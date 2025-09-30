# Distributed Training Communication Test

This is a quick test to verify if internode communication works for distributed training on MIT Supercloud with V100 GPUs.

## What it tests

- ✅ SLURM job setup with 2 nodes, 2 GPUs each (4 total GPUs)
- ✅ PyTorch distributed initialization with NCCL backend
- ✅ Basic network connectivity between nodes
- ✅ GPU availability and assignment
- ✅ Inter-GPU communication (all-reduce operation)
- ✅ Broadcast operations
- ✅ Node mapping to verify multi-node setup

## How to run

1. **Make sure you have a virtual environment set up:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch
   ```

2. **Submit the test job:**

   ```bash
   sbatch --nodes=2 test_distributed_comm.sh
   ```

3. **Check the results:**

   ```bash
   # Monitor the job
   squeue -u $USER
   
   # Check output when complete
   cat logs/test_dist_comm_[JOB_ID].out
   ```

## Expected output

If everything works correctly, you should see:

- ✅ All processes initialize successfully
- ✅ GPU detection and assignment
- ✅ Node mapping showing 2 different nodes
- ✅ Communication tests pass
- ✅ Final success message

## If the test fails

Common issues and solutions:

1. **Environment setup failure:**
   - Make sure virtual environment exists and has PyTorch installed
   - Check that CUDA modules are available

2. **Process group initialization failure:**
   - Network connectivity issues between nodes
   - Firewall blocking the master port (12355)
   - NCCL not properly configured

3. **GPU assignment issues:**
   - SLURM GPU allocation problems
   - Check if V100 GPUs are actually available

4. **Communication test failure:**
   - NCCL backend issues
   - Network interface problems
   - Try different NCCL_SOCKET_IFNAME settings

## Files created

- `test_distributed_comm.py` - Python test script
- `test_distributed_comm.sh` - SLURM submission script
- `logs/test_dist_comm_[JOB_ID].out` - Test output
- `logs/test_dist_comm_[JOB_ID].err` - Error log

## Quick interpretation

- **If test passes:** Your cluster supports distributed training! 🎉
- **If test fails:** There are communication issues that need to be resolved before running distributed training.

This test is specifically designed for 2 nodes with 2 V100 GPUs each. The test should complete in under 5 minutes if everything is working properly.
