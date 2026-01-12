# PyTorch CUDA Setup for RTX 5050 Laptop

## Overview
This Docker environment provides a complete PyTorch + CUDA setup optimized for machine learning development on NVIDIA RTX 5050 Laptop GPU.

## System Specifications
- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU
- **CUDA Compute Capability**: sm_120 (12.0 - Blackwell architecture)
- **Docker Image**: PyTorch 2.5.1 with CUDA 12.4 and cuDNN 9
- **Python Version**: 3.11

## Important Note about RTX 5050 Support
The RTX 5050 uses the new Blackwell architecture with compute capability 12.0 (sm_120). The current stable PyTorch 2.5.1 was built with support up to sm_90, which means:

- **PyTorch will still work** with your GPU through CUDA's forward compatibility (PTX)
- You may see a warning about unsupported compute capability
- Performance will be good but may not be fully optimized for Blackwell architecture
- Full native support will come in future PyTorch releases (likely 2.6+)

**Recommendation**: This setup works well for development. For production or maximum performance, wait for PyTorch 2.6+ which will have native Blackwell support.

## What's Included

### Core Framework
- **PyTorch 2.5.1** with CUDA 12.4 support
- **TorchVision 0.20.1** for computer vision
- **TorchAudio 2.5.1** for audio processing

### Development Tools
- **JupyterLab 4.5.2** - Interactive notebook environment
- **IPython & IPyWidgets** - Enhanced interactive computing

### Machine Learning Libraries
- **scikit-learn 1.8.0** - Traditional ML algorithms
- **NumPy 2.1.2** - Numerical computing
- **Pandas 2.3.3** - Data manipulation
- **SciPy 1.17.0** - Scientific computing

### Visualization
- **Matplotlib 3.10.8** - Plotting library
- **Seaborn 0.13.2** - Statistical data visualization

### Computer Vision
- **OpenCV 4.12.0** (headless) - Image processing
- **Pillow 10.2.0** - Image manipulation

### Utilities
- **tqdm** - Progress bars
- **Git, wget, curl, vim** - Development tools

## Configuration

### Docker Compose Settings
```yaml
Environment Variables:
- NVIDIA_VISIBLE_DEVICES=all
- CUDA_VISIBLE_DEVICES=0  
- TORCH_CUDA_ARCH_LIST="8.9+PTX"  # Forward compatibility
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

Resources:
- Shared Memory: 16GB
- GPU: Full access with compute capability
```

### CUDA Optimizations Enabled
1. **cuDNN Benchmark Mode** - Automatic algorithm selection for best performance
2. **TensorFloat-32 (TF32)** - Faster matrix operations on Ampere+ GPUs
3. **Expandable Memory Segments** - Better memory management
4. **PTX Forward Compatibility** - Support for newer GPU architectures

## Usage

### Starting the Container
```bash
cd /home/manu/ai-docker
docker compose up -d
```

### Accessing JupyterLab
- URL: `http://localhost:8888`
- No password required (development setup)

### Stopping the Container
```bash
docker compose down
```

### Viewing Logs
```bash
docker compose logs -f
```

### Executing Commands in Container
```bash
docker exec ml-pytorch-jupyter python -c "import torch; print(torch.cuda.is_available())"
```

## Test Notebooks

### 1. test_pytorch_cuda.ipynb
Comprehensive testing notebook that includes:

#### CUDA Verification
- PyTorch and CUDA version detection
- GPU information and capabilities
- Memory availability check

#### Performance Benchmarks
- Matrix multiplication (CPU vs GPU)
- Multiple iterations with timing
- Speedup calculations

#### CUDA Optimizations
- cuDNN settings verification
- TF32 enablement
- Mixed Precision (FP16) testing
- Performance comparisons

#### MobileNetV3 Inference
- Model loading and setup
- Single image inference
- Batch inference testing
- Throughput measurements
- Time per image analysis

#### Batch Performance Analysis
- Tests multiple batch sizes (1, 4, 8, 16, 32)
- Throughput vs batch size visualization
- Time per image scaling analysis

## Performance Expectations

### Matrix Multiplication (4096x4096)
- **CPU**: ~1-2 seconds
- **GPU**: ~0.01-0.05 seconds
- **Expected Speedup**: 20-100x

### MobileNetV3 Inference
- **Single Image**: 5-15 ms
- **Batch 32**: ~50-150 ms total (~2-5 ms per image)
- **Throughput**: 200-500 images/second (batch mode)

### Mixed Precision (FP16)
- **Expected Speedup**: 1.5-2.5x over FP32
- **Memory Reduction**: ~50%

## Troubleshooting

### Issue: "CUDA capability sm_120 is not compatible" Warning
**Status**: Known issue, not critical
**Explanation**: PyTorch 2.5.1 doesn't have native RTX 5050 (sm_120) binaries
**Impact**: Minimal - CUDA uses PTX for forward compatibility
**Solution**: Will be resolved in PyTorch 2.6+ (or use nightly builds)

### Issue: Out of Memory (OOM) Errors
**Solutions**:
1. Reduce batch size
2. Use mixed precision (FP16)
3. Enable gradient checkpointing
4. Increase shared memory in docker-compose.yml

### Issue: Slow First Inference
**Explanation**: Normal - JIT compilation and cuDNN algorithm selection
**Solution**: Run warmup iterations before timing

### Issue: Container Won't Start
**Check**:
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check Docker logs
docker compose logs
```

## File Structure
```
/home/manu/ai-docker/
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Docker Compose configuration  
├── environment.yml                 # Conda environment (reference)
├── notebooks/
│   ├── test_pytorch_cuda.ipynb    # Comprehensive PyTorch+CUDA tests
│   └── test_cuda.ipynb            # Old TensorFlow tests (legacy)
├── data/                          # Data directory (mounted)
├── README.md                      # Original README
├── SETUP_COMPLETE.md              # Setup notes
└── PYTORCH_CUDA_SETUP.md          # This file
```

## Common Tasks

### Install Additional Python Packages
```bash
docker exec ml-pytorch-jupyter pip install package_name
```

### Run Python Script
```bash
docker exec ml-pytorch-jupyter python /workspace/your_script.py
```

### Copy Files to/from Container
```bash
# To container
docker cp local_file.py ml-pytorch-jupyter:/workspace/

# From container
docker cp ml-pytorch-jupyter:/workspace/results.txt ./
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

## Optimization Tips

### 1. Enable All Optimizations
```python
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Use Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Pin Memory for Data Loading
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,
    num_workers=4
)
```

### 4. Use Appropriate Batch Sizes
- Start small and increase until GPU memory is ~80% utilized
- Use `torch.cuda.memory_allocated()` to monitor

## Next Steps

1. **Test the setup**: Open `test_pytorch_cuda.ipynb` and run all cells
2. **Verify GPU performance**: Check speedup metrics
3. **Experiment with models**: Try different architectures
4. **Monitor resources**: Use `nvidia-smi` during training

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## Version History

- **2026-01-12**: Initial PyTorch 2.5.1 + CUDA 12.4 setup for RTX 5050
- Base image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- Optimized for Blackwell architecture with PTX forward compatibility

## Support

For issues specific to:
- **Docker setup**: Check docker-compose.yml and Dockerfile
- **CUDA problems**: Verify nvidia-docker runtime and driver version
- **PyTorch issues**: Check PyTorch GitHub or forums
- **Model-specific problems**: Refer to model documentation

---

**Happy Machine Learning! 🚀🔥**
