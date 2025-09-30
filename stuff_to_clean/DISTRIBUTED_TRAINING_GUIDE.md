# Distributed GRPO Training Guide

This guide explains how to use the new distributed training setup for GRPO code execution training with 2 V100 GPUs on MIT Supercloud.

## 🚀 Quick Start

### Submit distributed training job

```bash
sbatch submit_grpo_training_distributed.sh
```

### With config overrides

```bash
sbatch submit_grpo_training_distributed.sh --training_args.max_steps=100 --training_args.learning_rate=0.0001
```

## 📋 What's New

### Performance Improvements

- **2x faster training**: Uses both V100 GPUs simultaneously
- **Optimized batch sizes**: Larger effective batch sizes for better convergence
- **Memory efficient**: Distributed data parallel reduces per-GPU memory usage
- **Smart evaluation**: Only runs on main process to avoid duplication

### Key Features

1. **Distributed Data Parallel (DDP)**: Splits batches across GPUs
2. **Optimized for V100**: Memory-efficient settings for V100 GPUs
3. **Single-node setup**: Perfect for MIT Supercloud's 2-GPU nodes
4. **Automatic scaling**: Batch sizes auto-adjust for distributed training

## 🔧 Technical Details

### SLURM Configuration

- **GPUs**: 2 V100s (`--gres=gpu:volta:2`)
- **Tasks**: 2 processes (`--ntasks=2`)
- **Memory**: 64GB (32GB per GPU)
- **Launch method**: `torchrun` for distributed coordination

### Memory Optimization

| Model Size | Per-GPU Batch | Effective Batch | Memory Strategy |
|------------|---------------|-----------------|-----------------|
| 1.5B       | 4             | 8               | Moderate optimization |
| 3B         | 2             | 4               | Aggressive optimization |

### Distributed Architecture

```
Node 1 (2 V100s)
├── Rank 0 (GPU 0) - Main process
│   ├── Model replica
│   ├── W&B logging
│   └── MBPP evaluation
└── Rank 1 (GPU 1) - Worker process
    ├── Model replica
    └── Training only
```

## 📊 Expected Performance Gains

### Training Speed

- **Single GPU**: ~100 steps/hour
- **Distributed (2 GPUs)**: ~180-200 steps/hour
- **Speedup**: ~1.8-2x faster

### Memory Efficiency

- **Effective batch size**: 2x larger than single GPU
- **Per-GPU memory**: Reduced due to batch splitting
- **Model memory**: Shared across GPUs

## 🔍 Monitoring

### Log Files Location

```
logs/grpo_distributed_job_[JOB_ID]_[TIMESTAMP]/
├── training_output.log       # Combined training logs
├── job_summary.txt          # Job summary and stats
├── checkpoints/             # Model checkpoints
└── wandb/                   # W&B logs (if enabled)
```

### Key Metrics to Watch

- **Distributed metrics**: Rank info, world size
- **Memory usage**: Per-GPU memory consumption  
- **Training speed**: Steps per second
- **Convergence**: Effective batch size impact

## 🛠️ Troubleshooting

### Common Issues

#### 1. NCCL Communication Errors

```bash
# Check GPU visibility
nvidia-smi

# Verify network setup
echo $MASTER_ADDR
echo $MASTER_PORT
```

#### 2. Memory Issues

```bash
# Reduce batch size in config
--training_args.per_device_train_batch_size=2
```

#### 3. Process Hanging

```bash
# Check if all processes started
ps aux | grep grpo_code_execute_distributed
```

### Debug Mode

Add to SLURM script for detailed debugging:

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
```

## 🎯 Best Practices

### 1. Batch Size Tuning

- Start with recommended settings
- Monitor GPU memory usage
- Adjust per-device batch size if needed

### 2. Checkpointing

- Checkpoints saved only by main process
- Automatic backup to log directory
- Resume from checkpoint if job fails

### 3. Evaluation

- MBPP evaluation runs only on main process
- Consistent seeding across runs
- Results logged to W&B

### 4. Resource Management

- Use full 12-hour time limit for training
- Monitor job queue: `squeue -u $USER`
- Check GPU utilization: `watch nvidia-smi`

## 📈 Configuration Recommendations

### For Fast Iteration (Testing)

```bash
sbatch submit_grpo_training_distributed.sh \
  --training_args.max_steps=50 \
  --evaluation.eval_interval_steps=10 \
  --training_args.per_device_train_batch_size=2
```

### For Production Training

```bash
sbatch submit_grpo_training_distributed.sh \
  --training_args.max_steps=1000 \
  --evaluation.eval_interval_steps=100 \
  --training_args.learning_rate=0.00005
```

### For Memory-Constrained Models

```bash
sbatch submit_grpo_training_distributed.sh \
  --training_args.per_device_train_batch_size=1 \
  --training_args.gradient_accumulation_steps=4 \
  --training_args.gradient_checkpointing=true
```

## 🔬 Compared to Single GPU

| Aspect | Single GPU | Distributed (2 GPUs) |
|--------|------------|----------------------|
| Training Speed | 1x | ~1.8-2x |
| Effective Batch Size | 4-8 | 8-16 |
| Memory per GPU | High | Moderate |
| Setup Complexity | Simple | Moderate |
| Resource Usage | 50% | 100% |

## 🚨 Important Notes

1. **Single Node Only**: Optimized for 2 GPUs on one node
2. **V100 Specific**: Memory settings tuned for V100 GPUs
3. **Main Process**: Only rank 0 handles evaluation and logging
4. **Automatic Cleanup**: Distributed processes cleaned up on exit
5. **Backward Compatible**: Original single-GPU script still works

## 🎉 Success Indicators

You'll know it's working when you see:

- ✅ Both GPUs showing utilization in `nvidia-smi`
- ✅ Log messages showing "Rank 0" and "Rank 1"
- ✅ Faster steps/second compared to single GPU
- ✅ Larger effective batch sizes in logs
- ✅ NCCL communication success messages

## 📞 Support

If you encounter issues:

1. Check the job logs in the generated log directory
2. Verify GPU availability with `nvidia-smi`
3. Ensure your virtual environment has all required packages
4. Try reducing batch sizes if you see OOM errors

Happy distributed training! 🚀
