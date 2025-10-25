import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from triton.testing import do_bench
import sys
import os

from state_passing_orig import _state_passing_fwd
from parallel_state_passing import _state_passing_fwd_cp_dist

def gather_all_states(states: torch.Tensor, dA_chunk_cumsum: torch.Tensor, group: dist.ProcessGroup):
    world_size = group.size()
    rank = group.rank()
    batch, nchunks, nheads, dim = states.shape

    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)

    all_states = torch.empty((world_size, batch, nchunks, nheads, dim), device=states.device, dtype=states.dtype)
    dist.all_gather_into_tensor(all_states, states, group=group)
    all_states = all_states.permute(1, 0, 2, 3, 4).contiguous().view(batch, world_size * nchunks, nheads, dim)

    all_dA_chunk_cumsum = torch.empty((world_size, batch, nheads, nchunks), device=dA_chunk_cumsum.device, dtype=dA_chunk_cumsum.dtype)
    dist.all_gather_into_tensor(all_dA_chunk_cumsum, dA_chunk_cumsum, group=group)
    all_dA_chunk_cumsum = all_dA_chunk_cumsum.permute(1, 2, 0, 3).contiguous().view(batch, nheads, world_size * nchunks)

    return all_states, all_dA_chunk_cumsum

def ref_slow_allgather(states: torch.Tensor, dA_chunk_cumsum: torch.Tensor, initial_states: torch.Tensor, group: dist.ProcessGroup):
    world_size = group.size()
    rank = group.rank()
    batch, nchunks, nheads, dim = states.shape

    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)

    all_states, all_dA_chunk_cumsum = gather_all_states(states, dA_chunk_cumsum, group)

    all_states = torch.empty((world_size, batch, nchunks, nheads, dim), device=states.device, dtype=states.dtype)
    dist.all_gather_into_tensor(all_states, states, group=group)
    all_states = all_states.permute(1, 0, 2, 3, 4).contiguous().view(batch, world_size * nchunks, nheads, dim)

    all_dA_chunk_cumsum = torch.empty((world_size, batch, nheads, nchunks), device=dA_chunk_cumsum.device, dtype=dA_chunk_cumsum.dtype)
    dist.all_gather_into_tensor(all_dA_chunk_cumsum, dA_chunk_cumsum, group=group)
    all_dA_chunk_cumsum = all_dA_chunk_cumsum.permute(1, 2, 0, 3).contiguous().view(batch, nheads, world_size * nchunks)

    output, final_states = _state_passing_fwd(states=all_states, dA_chunk_cumsum=all_dA_chunk_cumsum, initial_states=initial_states)

    if rank != world_size - 1:
        final_states = output[:, (rank + 1) * nchunks, :, :]

    return output[:, rank * nchunks:(rank + 1) * nchunks, :, :], final_states

def check_correctness(states: torch.Tensor, dA_chunk_cumsum: torch.Tensor, initial_states: torch.Tensor, group: dist.ProcessGroup):
    rank = group.rank()
    world_size = group.size()
    batch, nchunks, nheads, dim = states.shape

    print(f"{rank=} Starting distributed state passing", file=sys.stderr)
    output_dist, final_states_dist = _state_passing_fwd_cp_dist(states, dA_chunk_cumsum, initial_states, group=dist.group.WORLD)

    print(f"{rank=} Starting ref state passing", file=sys.stderr)
    output_ref, final_states_ref = ref_slow_allgather(states, dA_chunk_cumsum, initial_states, dist.group.WORLD)

    first_states_tensor = torch.empty(world_size, batch, nheads, dim, device='cuda', dtype=torch.float)
    dist.all_gather_into_tensor(first_states_tensor, output_dist[:, 0, :, :].contiguous(), group=dist.group.WORLD)

    print(f"{rank=} Checking results", file=sys.stderr)

    # if rank != 0: return

    if final_states_ref is not None:
        if not torch.allclose(final_states_ref, final_states_dist, atol=1e-6, rtol=1e-5):
            print(f'{rank=} bad final states', file=sys.stderr)
            print(f'{rank=}', final_states_ref.shape)
            print(f'{rank=}', final_states_ref[0, 0, :10])
            print(f'{rank=}', final_states_dist[0, 0, :10])
            print(f'{rank=} max diff', (final_states_ref - final_states_dist).abs().max())

        if rank < world_size - 1 and not torch.allclose(final_states_ref, first_states_tensor[rank + 1], atol=1e-6, rtol=1e-5):
            target = first_states_tensor[rank + 1]
            print(f'{rank=} bad initial states {(final_states_ref - target).abs().max()}', file=sys.stderr)
            print(f'{rank=}', final_states_ref[0, 0, :10])
            print(f'{rank=}', target[0, 0, :10])

    if not torch.allclose(output_ref, output_dist, atol=1e-6, rtol=1e-5):
        print(f'{rank=} bad output {(output_ref - output_dist).abs().max()}', file=sys.stderr)
        print(f'{rank=}', output_ref[0, :10, 0, :10])
        print(f'{rank=}', output_dist[0, :10, 0, :10])

def benchmark(states: torch.Tensor, dA_chunk_cumsum: torch.Tensor, initial_states: torch.Tensor, group: dist.ProcessGroup):
    all_states, all_dA_chunk_cumsum = gather_all_states(states, dA_chunk_cumsum, group)

    def run_ref():
        return _state_passing_fwd(states=all_states, dA_chunk_cumsum=all_dA_chunk_cumsum, initial_states=initial_states)

    def run_dist():
        return _state_passing_fwd_cp_dist(states, dA_chunk_cumsum, initial_states, group=dist.group.WORLD)

    ref_speed = do_bench(run_ref)
    dist_speed = do_bench(run_dist)

    print(f'Ref: {ref_speed}')
    print(f'Dist: {dist_speed}')


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(f"cuda:{rank}")
    dist.init_process_group("nccl")

    TOTAL_NCHUNKS = 1024
    BATCH = 4
    NHEADS = 8
    DIM = 256 * 256

    NCHUNKS = TOTAL_NCHUNKS // world_size

    states = torch.randn(BATCH, NCHUNKS, NHEADS, DIM, device='cuda', dtype=torch.float)
    dA_chunk_cumsum = torch.log(torch.sigmoid(torch.randn(BATCH, NHEADS, NCHUNKS, device='cuda', dtype=torch.float)))
    initial_states = torch.zeros(BATCH, NHEADS, DIM, device='cuda', dtype=torch.float)

    check_correctness(states, dA_chunk_cumsum, initial_states, group=dist.group.WORLD)
    benchmark(states, dA_chunk_cumsum, initial_states, group=dist.group.WORLD)



if __name__ == '__main__': 
    main()