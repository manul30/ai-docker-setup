"""
Configuration for MobileNetV3 Object Detection Training
"""

class Config:
    """Training configuration"""
    
    # Model configuration
    IMG_SIZE = 320  # Optimized for SSDLite320 (mobile-friendly)
    NUM_CLASSES = 2  # background + 1 object class
    
    # Training hyperparameters - QUICK TEST MODE 🚀
    NUM_EPOCHS = 20  # Quick testing (use 150 for production)
    BATCH_SIZE = 16   # Smaller batch for better generalization
    LEARNING_RATE = 0.002  # Higher initial LR with strong augmentation
    WEIGHT_DECAY = 0.0001  # Reduced for less regularization penalty
    MOMENTUM = 0.937  # YOLO's momentum value
    
    # Optimization techniques
    USE_MIXED_PRECISION = True
    ACCUMULATION_STEPS = 4  # Effective batch size = 64 (matches YOLO)
    GRADIENT_CLIP_VALUE = 10.0
    
    # Learning rate scheduler
    WARMUP_EPOCHS = 2  # Faster warmup for testing
    SCHEDULER_T_MAX = 18  # NUM_EPOCHS - WARMUP_EPOCHS
    
    # Quantization-Aware Training (QAT) - EARLY FOR TESTING
    ENABLE_QAT = True
    QAT_START_EPOCH = 12  # Start early for quick testing (use 120 for production)
    
    # Pruning - DISABLED (incompatible with QAT in same training run)
    ENABLE_PRUNING = False  # QAT + Pruning together causes dimension mismatches
    PRUNING_RATIO = 0.3  # 30% sparsity
    PRUNING_START_EPOCH = 16  # After QAT (use 130 for production)
    
    # Early stopping - SHORTER FOR TESTING
    EARLY_STOPPING_PATIENCE = 10  # Quick testing (use 30 for production)
    
    # Data augmentation (YOLO-style aggressive augmentation)
    USE_AUGMENTATION = True
    AUGMENTATION_PROB = 0.8  # 80% of images augmented
    
    # Paths - OBJECT DETECTION MODEL
    DATA_DIR = '/workspace/data'
    MODEL_SAVE_PATH = '/workspace/data/best_detection_mobilenetv3_ssd.pth'
    INT8_MODEL_PATH = '/workspace/data/detection_mobilenetv3_ssd_int8.pth'
    ANDROID_MODEL_PATH = '/workspace/data/detection_mobilenetv3_android.ptl'
    IOS_MODEL_PATH = '/workspace/data/detection_mobilenetv3_ios.mlmodel'
    ONNX_MODEL_PATH = '/workspace/data/detection_mobilenetv3_onnx.onnx'
    
    # Roboflow API
    ROBOFLOW_API_KEY = "LHiJvoAFmvmbSi50SwC1"
    ROBOFLOW_WORKSPACE = "hvac-whaik"
    ROBOFLOW_PROJECT = "ai-hvac-nameplate-focus-kcnb5"
    ROBOFLOW_VERSION = 11
    
    @classmethod
    def display(cls):
        """Display configuration"""
        print("="*80)
        print("🧪 QUICK TEST CONFIGURATION - OBJECT DETECTION")
        print("="*80)
        print(f"Model: MobileNetV3-Large + SSDLite (Object Detection)")
        print(f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE} (Effective: {cls.BATCH_SIZE * cls.ACCUMULATION_STEPS})")
        print(f"Epochs: {cls.NUM_EPOCHS} ⚡ QUICK TEST MODE")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"\nOptimizations:")
        print(f"  - Mixed Precision: {cls.USE_MIXED_PRECISION}")
        print(f"  - Gradient Accumulation: {cls.ACCUMULATION_STEPS}x")
        print(f"  - QAT: {cls.ENABLE_QAT} (starts at epoch {cls.QAT_START_EPOCH}) ⚡")
        print(f"  - Pruning: {cls.ENABLE_PRUNING} ({cls.PRUNING_RATIO*100:.0f}% at epoch {cls.PRUNING_START_EPOCH})")
        print(f"\n⚡ TEST MODE: QAT will trigger at epoch {cls.QAT_START_EPOCH}/{cls.NUM_EPOCHS}")
        print(f"💡 For production, set NUM_EPOCHS=150, QAT_START_EPOCH=120")
        print("="*80 + "\n")
