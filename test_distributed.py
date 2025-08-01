#!/usr/bin/env python3
"""
Simple test script to verify distributed training setup works correctly.
"""

import os
import torch
import torch.distributed as dist

def test_distributed():
    """Test if distributed training environment is set up correctly."""
    
    print("🧪 Testing Distributed Training Setup")
    print("=" * 50)
    
    # Check environment variables
    world_size = os.environ.get('WORLD_SIZE', '1')
    rank = os.environ.get('RANK', '0')
    local_rank = os.environ.get('LOCAL_RANK', '0')
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    print(f"Environment variables:")
    print(f"  WORLD_SIZE: {world_size}")
    print(f"  RANK: {rank}")
    print(f"  LOCAL_RANK: {local_rank}")
    print(f"  MASTER_ADDR: {master_addr}")
    print(f"  MASTER_PORT: {master_port}")
    print()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ CUDA not available")
        return False
    
    print()
    
    # Test distributed initialization if in distributed mode
    if int(world_size) > 1:
        try:
            print("🌐 Initializing distributed training...")
            
            # Set device
            local_rank = int(local_rank)
            torch.cuda.set_device(local_rank)
            
            # Initialize process group
            dist.init_process_group(backend='nccl')
            
            # Get distributed info
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            print(f"✅ Distributed training initialized successfully!")
            print(f"   Process rank: {rank}/{world_size}")
            print(f"   Local device: cuda:{local_rank}")
            print(f"   Device name: {torch.cuda.get_device_name(local_rank)}")
            
            # Test all-reduce operation
            print("\n🔄 Testing all-reduce communication...")
            tensor = torch.tensor([rank], dtype=torch.float32, device=f'cuda:{local_rank}')
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected_sum = sum(range(world_size))
            
            if tensor.item() == expected_sum:
                print(f"✅ All-reduce test passed! Sum: {tensor.item()}")
            else:
                print(f"❌ All-reduce test failed! Expected {expected_sum}, got {tensor.item()}")
                return False
            
            # Only main process prints final status
            if rank == 0:
                print(f"\n🎉 Distributed setup verification complete!")
                print(f"   Ready for training on {world_size} GPUs")
            
            # Cleanup
            dist.destroy_process_group()
            
        except Exception as e:
            print(f"❌ Distributed initialization failed: {e}")
            return False
    else:
        print("📱 Single GPU mode - no distributed setup needed")
    
    return True

if __name__ == "__main__":
    success = test_distributed()
    exit(0 if success else 1)