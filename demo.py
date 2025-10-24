import os
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(f"cuda:{rank}")
dist.init_process_group("nccl")

prev_rank = (rank - 1) % world_size
next_rank = (rank + 1) % world_size

# Allocate a tensor
t = symm_mem.empty(4096, device="cuda")

print("Allocated a tensor!")

# Establish symmetric memory and obtain the handle
hdl = symm_mem.rendezvous(t, dist.group.WORLD)

print("Finished rdzv")

peer_buf = hdl.get_buffer(next_rank, t.shape, t.dtype)

# Pull
t.fill_(rank)
hdl.barrier(channel=0)
pulled = torch.empty_like(t)
pulled.copy_(peer_buf)
hdl.barrier(channel=0)
assert pulled.eq(next_rank).all()

print("Finished pull!")

# Push
hdl.barrier(channel=0)
to_push = torch.full_like(t, rank)
peer_buf.copy_(to_push)
hdl.barrier(channel=0)
assert t.eq(prev_rank).all()

# Direct
t.fill_(rank)
hdl.barrier(channel=0)
torch.add(peer_buf, rank, out=peer_buf)
hdl.barrier(channel=0)
assert t.eq(rank + prev_rank).all()

# Raw pointers for CUDA/Triton kernels
hdl.buffer_ptrs
hdl.multicast_ptr
hdl.signal_pad_ptrs

print("Finished everything!")