"""
Model export functions for mobile deployment
"""
import torch
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
import coremltools as ct
import onnx
from onnxsim import simplify


def export_to_android(model, config):
    """Export model to TorchScript Mobile for Android"""
    print("="*80)
    print("🤖 ANDROID EXPORT - TorchScript Mobile")
    print("="*80 + "\n")
    
    # Prepare model for export
    model_export = model.cpu()
    model_export.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    
    # Trace the model
    print("Tracing model for TorchScript...")
    traced_model = torch.jit.trace(model_export, example_input)
    
    # Optimize for mobile
    print("Optimizing for mobile deployment...")
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save TorchScript Mobile model
    optimized_model._save_for_lite_interpreter(config.ANDROID_MODEL_PATH)
    
    print(f"✓ Android model saved to: {config.ANDROID_MODEL_PATH}")
    print(f"  File size: {os.path.getsize(config.ANDROID_MODEL_PATH) / (1024*1024):.2f} MB")
    
    # Test inference
    print("\nTesting Android model inference...")
    with torch.no_grad():
        output = optimized_model(example_input)
        print(f"✓ Output shape: {output[0]['boxes'].shape if 'boxes' in output[0] else 'Detection format'}")
    
    print("\n" + "="*80)
    return optimized_model


def export_to_ios(model, config):
    """Export model to CoreML for iOS"""
    print("="*80)
    print("🍎 iOS EXPORT - CoreML")
    print("="*80 + "\n")
    
    # Prepare model for CoreML export
    model_export = model.cpu()
    model_export.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    
    # Trace the model
    print("Tracing model for CoreML...")
    traced_model = torch.jit.trace(model_export, example_input)
    
    # Convert to CoreML
    print("Converting to CoreML format...")
    print("⚠ Note: This may take a few minutes...")
    
    # Define input shape
    input_shape = ct.Shape(shape=(1, 3, config.IMG_SIZE, config.IMG_SIZE))
    
    # Convert with optimizations
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=input_shape)],
        convert_to="mlprogram",  # Use ML Program format (iOS 15+)
        compute_precision=ct.precision.FLOAT16,  # Use FP16 for smaller size and faster inference
        minimum_deployment_target=ct.target.iOS15,
    )
    
    # Add metadata
    coreml_model.author = "AI HVAC Nameplate Detection"
    coreml_model.short_description = "MobileNetV3-SSD optimized for nameplate detection"
    coreml_model.version = "1.0"
    
    # Save CoreML model
    coreml_model.save(config.IOS_MODEL_PATH)
    
    print(f"\n✓ iOS model saved to: {config.IOS_MODEL_PATH}")
    print(f"  File size: {os.path.getsize(config.IOS_MODEL_PATH) / (1024*1024):.2f} MB")
    print(f"  Format: ML Program (iOS 15+)")
    print(f"  Precision: FP16 (optimized)")
    
    # Model info
    print(f"\nModel Input:")
    print(f"  Name: input")
    print(f"  Shape: [1, 3, {config.IMG_SIZE}, {config.IMG_SIZE}]")
    print(f"  Type: Image (RGB)")
    
    print("\n" + "="*80)
    return coreml_model


def export_to_onnx(model, config):
    """Export model to ONNX format"""
    print("="*80)
    print("🌐 ONNX EXPORT - Universal Format")
    print("="*80 + "\n")
    
    # Prepare model
    model_export = model.cpu()
    model_export.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    
    # Export to ONNX
    onnx_path = config.ONNX_MODEL_PATH.replace('_simplified', '')
    print("Exporting to ONNX format...")
    
    torch.onnx.export(
        model_export,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Simplify ONNX model
    print("Simplifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    simplified_model, check = simplify(onnx_model)
    
    if check:
        onnx.save(simplified_model, config.ONNX_MODEL_PATH)
        print(f"✓ Simplified ONNX model saved to: {config.ONNX_MODEL_PATH}")
        print(f"  File size: {os.path.getsize(config.ONNX_MODEL_PATH) / (1024*1024):.2f} MB")
    else:
        print("⚠ ONNX simplification failed, using original model")
    
    print(f"\n✓ ONNX model saved to: {onnx_path}")
    print(f"  File size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
    print(f"  Opset version: 13")
    print(f"  Dynamic batch size: Enabled")
    
    print("\n" + "="*80)
