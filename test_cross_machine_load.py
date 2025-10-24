"""
This module implements a Triton kernel for one-shot all-reduce.
This kernel performs an all-reduce operation on a Torch symmetric memory tensor distributed across
multiple devices. According to benchmark results, one-shot all reduce outperforms NCCL ring reduce
for small message sizes (<~400KB on a 8xH100 system with NVSwitch).
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

import os

from kraken import _ptx_utils as ptx_utils


@triton.jit
def ping_pong(
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    # Synchronize blocks with matching block_id across all participating devices before starting.
    # This ensures that all previous memory operations are visible.

    off = tl.program_id(0)

    tl.store(buf_tuple[0] + off, rank + 1)

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True
    )

    if buf_tuple[1] is not None:
        other = tl.load(buf_tuple[1] + off)
    else:
        other = 0

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True, hasPreviousMemAccess=True,
    )

    tl.store(buf_tuple[0] + off, other + 10 * (rank + 1))

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True, hasPreviousMemAccess=True,
    )

    if buf_tuple[1] is not None:
        other = tl.load(buf_tuple[1] + off)
    else:
        other = 0

    final = other + 100 * (rank + 1)

    # Synchronize all participating devices after the reduction is complete.
    # Subsequent kernel cannot overwrite the symmetric memory buffer until all devices reach this point.
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )

    tl.store(output_ptr + off, final)


def one_shot_all_reduce(group) -> torch.Tensor:
    NUM_BLOCKS = 100
    tensor = symm_mem.empty(NUM_BLOCKS, device='cuda', dtype=torch.int32)
    symm_mem_hdl = symm_mem.rendezvous(tensor, group=group)
    output = torch.empty_like(tensor)

    # Get the buffer pointers for each rank from the symmetric memory handle, and pass them as a tuple to the triton kernel.
    buf_tuple = tuple(
        symm_mem_hdl.get_buffer(group.rank() - i, tuple(tensor.shape), tensor.dtype) 
        if i <= group.rank() else None
        for i in range(2)
    )

    # symm_mem_hdl.signal_pad_ptrs_dev: An array of pointers pointing to signal_pads for each rank.
    # A signal pad is a memory region used for synchronization between devices.
    # `symm_mem_sync` kernel uses these signal pads to implement a cross-device barrier to ensure memory visibility of symmetric memory tensors.
    ping_pong[(NUM_BLOCKS, 1, 1)](
        buf_tuple,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        numel=tensor.numel(),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
    )

    return output


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(f"cuda:{rank}")
    dist.init_process_group("nccl")


    result = one_shot_all_reduce(group=dist.group.WORLD)

    print(f'{rank=} {result}')

if __name__ == '__main__': 
    main()