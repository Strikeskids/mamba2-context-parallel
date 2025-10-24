import math
import torch
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem

import triton
import triton.language as tl

from kraken import _ptx_utils as ptx_utils

MIN_BLOCK_SIZE = 64

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['dim'],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr, out_ptr, final_states_ptr, dA_cs_ptr, initstates_ptr, seq_idx_ptr,
    signal_pad_ptrs, state_comm_ptrs, dA_comm_ptrs,
    # Matrix dimensions
    dim, nchunks, seqlen, chunk_size,
    # Strides
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_dim,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_out_dim,
    stride_final_states_batch, stride_final_states_head, stride_final_states_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head,
    stride_initstates_batch, stride_initstates_head, stride_initstates_dim,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_state_comm_db, stride_state_comm_batch, stride_state_comm_head, stride_state_comm_dim,
    stride_dA_comm_db, stride_dA_comm_block, stride_dA_comm_batch, stride_dA_comm_head,
    # distributed parameters
    rank,
    # Meta-parameters
    WORLD_SIZE_BITS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head
    state_comm_offsets = pid_b * stride_state_comm_batch + pid_h * stride_state_comm_head
    dA_comm_offsets = pid_b * stride_dA_comm_batch + pid_h * stride_dA_comm_head

    # if HAS_SEQ_IDX:
    #     seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim
    dA_comm_offsets += pid_m * stride_dA_comm_block
    state_comm_offsets += offs_m * stride_state_comm_dim

    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    # seq_idx = 0

    # our monoid is over pairs (state, dA_sum)
    # the update function is
    # (s1, a1) o (s2, a2) = (exp(a2) * s1 + s2, a1 + a2)

    dA_sum = 0

    # first, compute the final chunk value
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        dA_sum += dA_cs
        scale = tl.exp(dA_cs)
        # if HAS_SEQ_IDX:
        #     seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
        #     scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
        #     seq_idx = seq_idx_new
        states = scale * states + new_states
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk

    # store the local states at the end of the reduction to the communication buffer
    # we don't use a mask because this is intermediate data
    tl.store(state_comm_ptrs[rank] + state_comm_offsets, states, mask=None)
    tl.store(dA_comm_ptrs[rank] + dA_comm_offsets, dA_sum)

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, WORLD_SIZE, hasSubsequentMemAccess=True
    )

    comm_parity = 0

    # then, do the parallel scan to reduce the values together
    for prev_rank in tl.static_range(WORLD_SIZE_BITS):
        mask = offs_m < dim
        peer_rank = rank - (1 << prev_rank)
        if peer_rank < 0:
            break

        scale = tl.exp(dA_sum)
        peer_states = tl.load(state_comm_ptrs[peer_rank] + state_comm_offsets + comm_parity * stride_state_comm_db)
        states = peer_states * scale + states

        dA_sum += tl.load(dA_comm_ptrs[peer_rank] + dA_comm_offsets + comm_parity * stride_dA_comm_db)
        tl.store(state_comm_ptrs[rank] + state_comm_offsets + (1 - comm_parity) * stride_state_comm_db, states, mask=mask)
        tl.store(dA_comm_ptrs[rank] + dA_comm_offsets + (1 - comm_parity) * stride_dA_comm_db, dA_sum)

        ptx_utils.symm_mem_sync(
            signal_pad_ptrs, None, rank, WORLD_SIZE, hasPreviousMemAccess=True, hasSubsequentMemAccess=True
        )

        comm_parity = 1 - comm_parity

    # finally, redo the chunk-wise computation to write the outputs

    # load the state, either from initial states or from the communication buffer
    if rank == 0:
        if not HAS_INITSTATES:
            states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        else:
            initstates_ptrs = initstates_ptr + offs_m * stride_initstates_dim
            states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    else:
        # load from the previous computed total state
        states = tl.load(state_comm_ptrs[rank - 1] + state_comm_offsets + comm_parity * stride_state_comm_db).to(tl.float32)

    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk

    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        # if HAS_SEQ_IDX:
        #     seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
        #     scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
        #     seq_idx = seq_idx_new
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk



def _state_passing_fwd_cp_dist(states, dA_chunk_cumsum, initial_states=None, out_dtype=None, group=None):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, dim)
    out_dtype = states.dtype if out_dtype is None else out_dtype

    assert dA_chunk_cumsum.dtype == states.dtype

    state_comm_shape = (2, batch, nheads, dim)
    # we need to have this extra dimension because each block might not do the reduction at the same time
    dA_comm_shape = (2, triton.cdiv(dim, MIN_BLOCK_SIZE), batch, nheads)

    total_state_size = torch.prod(torch.tensor(state_comm_shape)).item() + torch.prod(torch.tensor(dA_comm_shape)).item()

    full_state_comm = symm_mem.empty(total_state_size, device=states.device, dtype=states.dtype)
    symm_mem_hdl = symm_mem.rendezvous(full_state_comm, group)

    world_size = group.size()

    state_comm_ptrs = tuple(
        symm_mem_hdl.get_buffer(i, tuple(state_comm_shape), full_state_comm.dtype, offset=0)
        for i in range(world_size)
    )

    dA_comm_ptrs = tuple(
        symm_mem_hdl.get_buffer(i, tuple(dA_comm_shape), full_state_comm.dtype, offset=0)
        for i in range(world_size)
    )

    out = torch.empty((batch, nchunks, nheads, dim), device=states.device, dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim), device=states.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states, out, final_states, dA_chunk_cumsum, initial_states, None,
            symm_mem_hdl.signal_pad_ptrs_dev, state_comm_ptrs,
            dim, nchunks,  0,  0,
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            final_states.stride(0), final_states.stride(1), final_states.stride(2),
            dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
            *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2))
              if initial_states is not None else (0, 0, 0)),
            *(0, 0),
            state_comm_ptrs[0].stride(0), state_comm_ptrs[0].stride(1), state_comm_ptrs[0].stride(2), state_comm_ptrs[0].stride(3),
            dA_comm_ptrs[0].stride(0), dA_comm_ptrs[0].stride(1), dA_comm_ptrs[0].stride(2),
            rank=group.rank(), 
            WORLD_SIZE_BITS=math.ceil(math.log2(world_size)),
            WORLD_SIZE=world_size,
            HAS_INITSTATES=initial_states is not None,
        )
    return out, final_states

