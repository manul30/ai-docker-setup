# Docker TensorFlow GPU Setup - Complete! ✓

## Successfully Configured

Your Docker environment with CUDA, TensorFlow, and XLA GPU acceleration is now fully operational.

### System Information
- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (Compute Capability 12.0 - Blackwell architecture)
- **Host CUDA**: 13.0
- **Driver**: 580.95.05
- **Container Base**: NVIDIA TensorFlow 25.01-tf2-py3 (Official NVIDIA NGC Container)
- **TensorFlow Version**: 2.17.0

### What's Working ✓
1. **GPU Detection**: TensorFlow successfully detects the RTX 5050
2. **GPU Computation**: Matrix operations execute successfully on GPU
3. **JupyterLab**: Running on http://localhost:8888/lab (no authentication required)
4. **Notebook Access**: Your notebooks are accessible in `/workspace/notebooks/`
5. **XLA Support**: Ready for XLA compilation and optimization

### Test Results
```
TensorFlow: 2.17.0
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
✓ GPU computation successful!
Result shape: (1000, 1000)
Result on device: /job:localhost/replica:0/task:0/device:GPU:0
```

GPU device created with 3503 MB memory available.

## Quick Start

### Access JupyterLab
Open your browser and go to:
```
http://localhost:8888/lab
```

No token required - authentication is disabled for convenience.

### Run Your Test Notebook
1. Navigate to `notebooks/test_cuda.ipynb`
2. The notebook includes comprehensive benchmarks:
   - CPU baseline
   - GPU baseline
   - XLA-GPU acceleration
   - XLA-CPU optimization

### Container Management

**Start the container:**
```bash
cd /home/manu/ai-docker
docker compose up -d
```

**Stop the container:**
```bash
docker compose down
```

**Rebuild after changes:**
```bash
docker compose down
docker compose up --build -d
```

**View logs:**
```bash
docker logs ml-jupyter
```

**Access container shell:**
```bash
docker exec -it ml-jupyter bash
```

## Key Configuration Files

### Dockerfile
- Base image: `nvcr.io/nvidia/tensorflow:25.01-tf2-py3`
- Additional packages: JupyterLab, tensorflow-datasets, keras
- Working directory: `/workspace`

### docker-compose.yml
- Runtime: nvidia (GPU passthrough)
- Port: 8888 (JupyterLab)
- Volumes:
  - `./notebooks:/workspace/notebooks` (your notebooks)
  - `./data:/workspace/data` (data directory)
- Shared memory: 16GB (for large models)

## Why This Works

The NVIDIA official TensorFlow container (`nvcr.io/nvidia/tensorflow:25.01-tf2-py3`) includes:
- Pre-compiled TensorFlow optimized for NVIDIA GPUs
- Native support for newer GPU architectures (including Blackwell)
- All required CUDA libraries properly configured
- Latest compatible CUDA/cuDNN versions

This eliminates the compatibility issues encountered with manual CUDA setup, especially for cutting-edge GPU architectures like your RTX 5050 (compute capability 12.0).

## Next Steps

1. **Run Benchmarks**: Execute `test_cuda.ipynb` to see GPU vs CPU performance
2. **Test XLA**: Compare XLA-compiled vs standard GPU execution
3. **Add Your Models**: Place your ML projects in the `notebooks/` directory
4. **Use GPU Memory Efficiently**: Monitor with `nvidia-smi` in the container

## Troubleshooting

**Check GPU visibility:**
```bash
docker exec ml-jupyter nvidia-smi
```

**Test TensorFlow GPU:**
```bash
docker exec ml-jupyter python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Monitor container logs:**
```bash
docker logs -f ml-jupyter
```

## Performance Tips

1. **Enable XLA compilation** for better performance:
   ```python
   import tensorflow as tf
   tf.config.optimizer.set_jit(True)
   ```

2. **Use mixed precision** for faster training:
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```

3. **Monitor GPU memory** to avoid OOM errors:
   ```python
   gpus = tf.config.list_physical_devices('GPU')
   tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

---

**Setup completed on**: 2026-01-11  
**Container status**: Running  
**JupyterLab**: Active on http://localhost:8888/lab
