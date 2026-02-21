# 🔨 Rebuild Instructions

## Changes Made

### 1. Fixed Model Creation Error
**Problem**: `AttributeError: 'SSDLiteFeatureExtractorMobileNet' object has no attribute 'out_channels'`

**Solution**: Updated `get_model()` function to correctly extract input channels from the SSD head instead of backbone.

### 2. Added Dependencies to Docker

Updated **Dockerfile** and **environment.yml** with all required packages:
- ✅ roboflow (dataset download)
- ✅ torch-pruning (model pruning)
- ✅ coremltools (iOS export)
- ✅ onnx (ONNX export)
- ✅ onnxsim (ONNX simplification)
- ✅ thop (FLOPs calculation)

### 3. Removed In-Notebook Installation
- ✅ Removed `!pip install` cell from notebook
- ✅ All dependencies now in Docker container

## 🚀 How to Rebuild

### Step 1: Rebuild Docker Container
```bash
cd /home/manu/ai-docker
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Step 2: Access Jupyter
```bash
# Get the container logs to find the URL
docker compose logs jupyter

# Or just open browser to:
# http://localhost:8888
```

### Step 3: Open Notebook
Navigate to: `notebooks/training_mobilenetv3_clean.ipynb`

## ✅ What's Fixed

### Before
```python
# ❌ This failed
in_channels = model.backbone.out_channels  # AttributeError!
```

### After
```python
# ✅ This works
in_channels = model.head.classification_head.module_list[0].in_channels
```

## �� New Docker Image Includes

All these packages are now pre-installed:
- PyTorch 2.9.1 + CUDA 12.8
- JupyterLab
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV (headless)
- Roboflow API client
- torch-pruning
- CoreML Tools
- ONNX + onnxsim
- THOP (PyTorch FLOPs counter)

## 🔍 Verification

After rebuild, run this in a notebook cell to verify:
```python
import torch
import roboflow
import torch_pruning
import coremltools
import onnx
import thop

print("✓ All packages installed!")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
```

## ⏱️ Rebuild Time

Expected: 5-10 minutes (depending on internet speed)

## 📝 Notes

- No need to install anything in the notebook anymore
- All dependencies are persistent in the Docker image
- The model creation now uses the correct API
- Training should work without errors

## 🐛 Troubleshooting

### If rebuild fails:
```bash
# Clean everything
docker compose down -v
docker system prune -af

# Rebuild
docker compose build --no-cache
docker compose up
```

### If packages still missing:
Check the Dockerfile has these lines:
```dockerfile
RUN pip install --no-cache-dir \
    roboflow \
    torch-pruning \
    coremltools \
    onnx \
    onnxsim \
    thop
```

## ✨ Ready to Train!

Once rebuilt, you can:
1. Open `training_mobilenetv3_clean.ipynb`
2. Run all cells from top to bottom
3. No installation steps needed
4. Everything should work smoothly

---

**Last Updated**: February 20, 2026
