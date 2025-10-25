import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from triton.testing import do_bench
import sys
import os

from state_passing_orig import _state_passing_fwd, _state_passing_fwd_persistent_limited_block

def check_correctness(states: torch.Tensor, dA_chunk_cumsum: torch.Tensor, initial_states: torch.Tensor, NUM_SMS: int):
    print("Starting persistent state passing", file=sys.stderr)
    output_dist, final_states_dist = \
        _state_passing_fwd_persistent_limited_block(states, dA_chunk_cumsum, initial_states, NUM_SMS=NUM_SMS)

    print("Starting ref state passing", file=sys.stderr)
    output_ref, final_states_ref = _state_passing_fwd(states, dA_chunk_cumsum, initial_states)

    print("Checking results", file=sys.stderr)

    if not torch.allclose(final_states_ref, final_states_dist, atol=1e-6, rtol=1e-5):
        print('bad final states', file=sys.stderr)
        print('', final_states_ref.shape)
        print('', final_states_ref[0, 0, :10])
        print('', final_states_dist[0, 0, :10])
        print('max diff', (final_states_ref - final_states_dist).abs().max())

    if not torch.allclose(output_ref, output_dist, atol=1e-6, rtol=1e-5):
        print(f'bad output {(output_ref - output_dist).abs().max()}', file=sys.stderr)
        print(f'', output_ref[0, :10, 0, :10])
        print(f'', output_dist[0, :10, 0, :10])

def benchmark(states: torch.Tensor, dA_chunk_cumsum: torch.Tensor, initial_states: torch.Tensor, NUM_SMS: int):
    def run_ref():
        return _state_passing_fwd(states, dA_chunk_cumsum, initial_states=initial_states)

    def run_persistent():
        return _state_passing_fwd_persistent_limited_block(states, dA_chunk_cumsum, initial_states, NUM_SMS=NUM_SMS)

    ref_speed = do_bench(run_ref)
    persistent_speed = do_bench(run_persistent)

    print(f'{NUM_SMS=}')
    print(f'Ref: {ref_speed}')
    print(f'Persistent: {persistent_speed}')


def main():
    NCHUNKS = 1024
    BATCH = 4
    NHEADS = 8
    DIM = 256 * 256

    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count * 4

    states = torch.randn(BATCH, NCHUNKS, NHEADS, DIM, device='cuda', dtype=torch.float)
    dA_chunk_cumsum = torch.log(torch.sigmoid(torch.randn(BATCH, NHEADS, NCHUNKS, device='cuda', dtype=torch.float)))
    initial_states = torch.zeros(BATCH, NHEADS, DIM, device='cuda', dtype=torch.float)

    check_correctness(states, dA_chunk_cumsum, initial_states, NUM_SMS=NUM_SMS)
    benchmark(states, dA_chunk_cumsum, initial_states, NUM_SMS=NUM_SMS)



if __name__ == '__main__': 
    main()