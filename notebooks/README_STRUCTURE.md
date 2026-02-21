# MobileNetV3 Training - Modular Structure

## 📁 File Structure

```
notebooks/
├── training_mobilenetv3_clean.ipynb    # ✨ NEW: Lightweight notebook
├── training_mobilenetv3.ipynb          # Original (full version)
├── config.py                           # Configuration class
├── scheduler.py                        # WarmupCosineLR scheduler
├── training_utils.py                   # Training functions
├── optimization_utils.py               # QAT & Pruning
├── export_utils.py                     # Mobile export
├── benchmark_utils.py                  # Benchmarking
├── DEPLOYMENT_EXAMPLES.md              # Deployment code
├── CHANGELOG_MOBILENETV3.md            # Change log
└── PRODUCTION_IMPROVEMENTS.md          # Full documentation
```

## 🚀 Quick Start

### Option 1: Use Lightweight Notebook (Recommended)

```bash
jupyter notebook training_mobilenetv3_clean.ipynb
```

**Benefits:**
- ✅ Only 16 cells (vs 45 in original)
- ✅ Fast to load in browser
- ✅ Clean and easy to follow
- ✅ All functionality preserved
- ✅ Easy to modify and debug

### Option 2: Use Full Notebook

```bash
jupyter notebook training_mobilenetv3.ipynb
```

**When to use:**
- You want inline documentation
- You need to see all implementation details
- You're learning how everything works

## 📝 Python Modules

### config.py
Contains all training configuration:
```python
from config import Config

config = Config()
config.display()

# Modify settings
config.IMG_SIZE = 224
config.BATCH_SIZE = 16
```

### scheduler.py
Custom learning rate scheduler:
```python
from scheduler import WarmupCosineLR

scheduler = WarmupCosineLR(optimizer, warmup_epochs=5, total_epochs=100)
scheduler.step()  # Call after each epoch
```

### training_utils.py
Training and validation functions:
```python
from training_utils import train_one_epoch, validate

train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
val_loss = validate(model, val_loader, device)
```

### optimization_utils.py
Model optimization (QAT & Pruning):
```python
from optimization_utils import apply_pruning, enable_qat, convert_to_int8

# Apply pruning
model = apply_pruning(model, config, device)

# Enable QAT
model = enable_qat(model, device)

# Convert to INT8
model_int8 = convert_to_int8(model, save_path)
```

### export_utils.py
Export to mobile formats:
```python
from export_utils import export_to_android, export_to_ios, export_to_onnx

export_to_android(model, config)
export_to_ios(model, config)
export_to_onnx(model, config)
```

### benchmark_utils.py
Performance benchmarking:
```python
from benchmark_utils import benchmark_model_size, benchmark_inference_speed

benchmark_model_size(fp32_path, int8_path, config)
benchmark_inference_speed(model_fp32, model_int8, config, device)
```

## 🎯 Training Workflow

### 1. Configure Training
Edit `config.py` to adjust hyperparameters:
```python
class Config:
    IMG_SIZE = 160
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    ENABLE_QAT = True
    QAT_START_EPOCH = 70
    ENABLE_PRUNING = True
    PRUNING_START_EPOCH = 80
```

### 2. Run Training
Open `training_mobilenetv3_clean.ipynb` and run all cells.

### 3. Training Process
```
Epoch 1-69:   Normal training
Epoch 70:     QAT enabled
Epoch 80:     Pruning applied
Epoch 81-100: Fine-tune with optimizations
```

### 4. Outputs
```
/workspace/data/
├── best_segmentation_model_lightweight.pth  # FP32 + Pruned
├── model_int8_quantized.pth                 # INT8 Quantized
├── model_android.ptl                        # Android
├── model_ios.mlmodel                        # iOS
├── model_onnx_simplified.onnx               # ONNX
└── training_YYYYMMDD_HHMMSS.log            # Logs
```

## 🔧 Customization Examples

### Change Model Architecture
Edit the `get_model()` function in the notebook:
```python
def get_model(num_classes):
    # Use different backbone
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    # ... modify as needed
    return model
```

### Adjust Pruning Ratio
Edit `config.py`:
```python
PRUNING_RATIO = 0.5  # 50% sparsity (more aggressive)
```

### Disable Optimizations
```python
ENABLE_QAT = False
ENABLE_PRUNING = False
```

### Change Dataset
Edit notebook cell:
```python
project = rf.workspace("your-workspace").project("your-project")
dataset = project.version(1).download("coco")
```

## 📊 Benefits of Modular Structure

### Before (Original Notebook)
- ❌ 45 cells, 2100+ lines
- ❌ Slow to load in browser
- ❌ Hard to find specific code
- ❌ Difficult to reuse functions
- ❌ Can't import in other notebooks

### After (Modular Structure)
- ✅ 16 cells, ~300 lines
- ✅ Fast to load
- ✅ Clean and organized
- ✅ Reusable functions
- ✅ Can import modules anywhere

## 🐛 Troubleshooting

### "Module not found" error
Make sure you're in the `notebooks/` directory:
```bash
cd /home/manu/ai-docker/notebooks
jupyter notebook
```

### Notebook won't load
Use the clean version:
```bash
jupyter notebook training_mobilenetv3_clean.ipynb
```

### Import errors
Install all dependencies:
```bash
pip install torch torchvision roboflow torch-pruning coremltools onnx onnxsim thop
```

## 📚 Documentation

- **DEPLOYMENT_EXAMPLES.md**: Complete deployment code for Android/iOS/Python
- **CHANGELOG_MOBILENETV3.md**: All changes and features
- **PRODUCTION_IMPROVEMENTS.md**: Detailed optimization guide

## 🎓 Learning Path

1. **Beginners**: Start with `training_mobilenetv3_clean.ipynb`
2. **Intermediate**: Read the Python modules to understand implementation
3. **Advanced**: Modify `config.py` and Python modules for custom experiments
4. **Deployment**: Follow `DEPLOYMENT_EXAMPLES.md`

## 💡 Tips

- Use the clean notebook for training
- Keep the original notebook as reference
- Modify Python modules for custom behavior
- All modules have proper logging
- Check logs in `/workspace/data/training_*.log`

## ✅ Recommended Workflow

```bash
# 1. Configure
vi config.py  # Adjust hyperparameters

# 2. Train
jupyter notebook training_mobilenetv3_clean.ipynb  # Run all cells

# 3. Deploy
# Follow DEPLOYMENT_EXAMPLES.md for your platform

# 4. Debug (if needed)
tail -f /workspace/data/training_*.log  # Monitor logs
```

## 🚀 Next Steps

1. Run the clean notebook to verify everything works
2. Experiment with different configurations in `config.py`
3. Deploy to your target platform using `DEPLOYMENT_EXAMPLES.md`
4. Monitor performance and iterate

---

**Questions?** Check the documentation files or the original notebook for detailed explanations.
