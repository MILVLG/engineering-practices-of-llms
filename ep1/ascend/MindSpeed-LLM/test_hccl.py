import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="hccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)

    x = torch.ones(1, device="npu:{}".format(local_rank)) * rank
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] after all_reduce: {x.item()} (world_size={world_size})")

if __name__ == "__main__":
    main()
