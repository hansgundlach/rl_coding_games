#!/usr/bin/env python3
"""
Basic test for internode communication in distributed PyTorch training.
Tests if NCCL communication works between nodes on MIT Supercloud.
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import socket
from datetime import datetime


def setup_distributed():
    """Initialize distributed training environment"""

    # Get distributed training parameters from environment
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    print(f"[Rank {rank}] Setting up distributed training...")
    print(f"[Rank {rank}] Hostname: {socket.gethostname()}")
    print(
        f"[Rank {rank}] Local rank: {local_rank}, Global rank: {rank}, World size: {world_size}"
    )
    print(f"[Rank {rank}] Master: {master_addr}:{master_port}")

    # Initialize the process group
    try:
        dist.init_process_group(
            backend="nccl",  # Use NCCL for GPU communication
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.default_pg_timeout,
        )
        print(f"[Rank {rank}] ✅ Process group initialized successfully")
        return True
    except Exception as e:
        print(f"[Rank {rank}] ❌ Failed to initialize process group: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability and assignment"""
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[Rank {rank}] Testing GPU availability...")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] ❌ CUDA not available")
        return False

    # Check GPU count
    gpu_count = torch.cuda.device_count()
    print(f"[Rank {rank}] Available GPUs: {gpu_count}")

    # Set GPU device
    if local_rank >= gpu_count:
        print(f"[Rank {rank}] ❌ Local rank {local_rank} >= available GPUs {gpu_count}")
        return False

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)

    print(f"[Rank {rank}] Using GPU {device}: {gpu_name}")
    print(
        f"[Rank {rank}] GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
    )

    return True


def test_basic_communication():
    """Test basic tensor communication between processes"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    print(f"[Rank {rank}] Testing basic communication...")

    # Create a test tensor on GPU
    tensor = torch.tensor([rank + 1.0], device=device)
    original_value = tensor.item()

    print(f"[Rank {rank}] Original tensor value: {original_value}")

    try:
        # Test all_reduce operation (sum across all processes)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced_value = tensor.item()

        # Expected sum: 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
        expected_sum = sum(range(1, world_size + 1))

        print(
            f"[Rank {rank}] All-reduce result: {reduced_value}, Expected: {expected_sum}"
        )

        if abs(reduced_value - expected_sum) < 1e-6:
            print(f"[Rank {rank}] ✅ All-reduce test PASSED")
            return True
        else:
            print(f"[Rank {rank}] ❌ All-reduce test FAILED")
            return False

    except Exception as e:
        print(f"[Rank {rank}] ❌ Communication test failed: {e}")
        return False


def test_broadcast():
    """Test broadcast operation"""
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    print(f"[Rank {rank}] Testing broadcast...")

    try:
        # Create tensor (only rank 0 has meaningful data)
        if rank == 0:
            tensor = torch.tensor([42.0, 7.0, 13.0], device=device)
            print(f"[Rank {rank}] Broadcasting tensor: {tensor.tolist()}")
        else:
            tensor = torch.zeros(3, device=device)
            print(f"[Rank {rank}] Waiting for broadcast...")

        # Broadcast from rank 0
        dist.broadcast(tensor, src=0)

        expected = [42.0, 7.0, 13.0]
        result = tensor.tolist()

        print(f"[Rank {rank}] Received tensor: {result}")

        if result == expected:
            print(f"[Rank {rank}] ✅ Broadcast test PASSED")
            return True
        else:
            print(f"[Rank {rank}] ❌ Broadcast test FAILED")
            return False

    except Exception as e:
        print(f"[Rank {rank}] ❌ Broadcast test failed: {e}")
        return False


def test_node_mapping():
    """Test which rank is on which node"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()

    print(f"[Rank {rank}] Testing node mapping...")

    # Gather hostnames from all ranks
    hostname_tensor = torch.zeros(
        world_size, 256, dtype=torch.uint8, device=torch.cuda.current_device()
    )

    # Convert hostname to tensor
    hostname_bytes = hostname.encode("utf-8")[:255]  # Limit to 255 chars
    hostname_tensor[rank, : len(hostname_bytes)] = torch.tensor(
        [b for b in hostname_bytes], dtype=torch.uint8
    )

    # All-reduce to gather all hostnames
    dist.all_reduce(hostname_tensor, op=dist.ReduceOp.SUM)

    # Extract and print node mapping
    node_map = {}
    for r in range(world_size):
        host_bytes = hostname_tensor[r].cpu().numpy()
        # Find end of string (first zero)
        end_idx = 0
        for i, b in enumerate(host_bytes):
            if b == 0:
                end_idx = i
                break
        else:
            end_idx = len(host_bytes)

        if end_idx > 0:
            host = bytes(host_bytes[:end_idx]).decode("utf-8")
            if host not in node_map:
                node_map[host] = []
            node_map[host].append(r)

    if rank == 0:
        print(f"[Rank {rank}] Node mapping:")
        for node, ranks in node_map.items():
            print(f"[Rank {rank}]   {node}: ranks {ranks}")

        num_nodes = len(node_map)
        print(f"[Rank {rank}] Total nodes: {num_nodes}")

        if num_nodes > 1:
            print(f"[Rank {rank}] ✅ Multi-node setup detected")
        else:
            print(
                f"[Rank {rank}] ⚠️ Single-node setup (no internode communication needed)"
            )

    return True


def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 DISTRIBUTED TRAINING COMMUNICATION TEST")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"Hostname: {socket.gethostname()}")

    # Check environment variables
    required_env = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    print(f"Environment variables:")
    for var in required_env:
        value = os.environ.get(var, "NOT SET")
        print(f"  {var}: {value}")

    if not all(os.environ.get(var) for var in required_env):
        print("❌ CRITICAL: Required environment variables not set!")
        print("This test must be run through SLURM with proper distributed setup.")
        sys.exit(1)

    print("\n" + "=" * 40)
    print("1️⃣ SETTING UP DISTRIBUTED ENVIRONMENT")
    print("=" * 40)

    if not setup_distributed():
        print("❌ CRITICAL: Failed to setup distributed environment")
        sys.exit(1)

    time.sleep(1)  # Give all processes time to initialize

    print("\n" + "=" * 40)
    print("2️⃣ TESTING GPU AVAILABILITY")
    print("=" * 40)

    if not test_gpu_availability():
        print("❌ CRITICAL: GPU test failed")
        dist.destroy_process_group()
        sys.exit(1)

    # Synchronize all processes
    dist.barrier()

    print("\n" + "=" * 40)
    print("3️⃣ TESTING NODE MAPPING")
    print("=" * 40)

    test_node_mapping()

    # Synchronize all processes
    dist.barrier()

    print("\n" + "=" * 40)
    print("4️⃣ TESTING BASIC COMMUNICATION")
    print("=" * 40)

    comm_success = test_basic_communication()

    # Synchronize all processes
    dist.barrier()

    print("\n" + "=" * 40)
    print("5️⃣ TESTING BROADCAST")
    print("=" * 40)

    broadcast_success = test_broadcast()

    # Synchronize all processes
    dist.barrier()

    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 60)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Total processes: {world_size}")
        print(f"Communication test: {'✅ PASS' if comm_success else '❌ FAIL'}")
        print(f"Broadcast test: {'✅ PASS' if broadcast_success else '❌ FAIL'}")

        if comm_success and broadcast_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Distributed training communication is working correctly.")
            print("✅ Internode communication is functional.")
            print("🚀 You can proceed with distributed training!")
        else:
            print("\n💥 SOME TESTS FAILED!")
            print("❌ Distributed training may not work properly.")
            print("🔍 Check network configuration and NCCL settings.")

    print(f"\n[Rank {rank}] Test completed at: {datetime.now()}")

    # Clean shutdown
    dist.destroy_process_group()

    if not (comm_success and broadcast_success):
        sys.exit(1)


if __name__ == "__main__":
    main()
