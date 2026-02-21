"""
Model optimization functions: QAT and Pruning
"""
import torch
import torch_pruning as tp
from torch.quantization import prepare_qat, convert
import logging

logger = logging.getLogger(__name__)


def apply_pruning(model, config, device):
    """Apply structured pruning to the model
    
    Note: Pruning with QAT is problematic because:
    1. QAT fake quantization modules track scale/zero_point per channel
    2. Pruning changes channel dimensions, breaking these tensors
    3. Solution: Skip pruning when QAT is enabled, or disable QAT first
    """
    print("\n" + "="*80)
    print(f"🔪 STRUCTURED PRUNING")
    print("="*80)
    logger.info(f"Checking pruning compatibility with QAT")
    
    # Check if QAT is enabled
    qat_was_enabled = False
    for name, module in model.named_modules():
        if 'fake_quant' in name.lower() or hasattr(module, 'weight_fake_quant'):
            qat_was_enabled = True
            break
    
    if qat_was_enabled:
        print("⚠️  QAT is currently enabled!")
        print("❌ Pruning is NOT compatible with QAT due to dimension mismatches")
        print("💡 Recommendation: Apply pruning BEFORE QAT, or disable pruning")
        print("\n🔄 SKIPPING PRUNING to avoid breaking QAT training...")
        logger.warning("Skipped pruning due to QAT incompatibility")
        print("="*80 + "\n")
        return model
    
    # Get initial model size
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Apply pruning
    example_inputs = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
    
    # Define importance metric (L1 norm)
    imp = tp.importance.MagnitudeImportance(p=1)
    
    # Configure pruner
    ignored_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            ignored_layers.append(module)
    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=config.PRUNING_RATIO,
        ignored_layers=ignored_layers,
    )
    
    # Prune the model
    pruner.step()
    
    final_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction = (initial_params - final_params) / initial_params * 100
    
    print(f"✓ Pruning complete:")
    print(f"  - Parameters before: {initial_params:,}")
    print(f"  - Parameters after:  {final_params:,}")
    print(f"  - Reduction: {reduction:.2f}%")
    logger.info(f"Pruning reduced parameters from {initial_params:,} to {final_params:,} ({reduction:.2f}% reduction)")
    
    print("="*80 + "\n")
    return model


def enable_qat(model, device):
    """Enable Quantization-Aware Training"""
    print("\n" + "="*80)
    print(f"🎯 ENABLING QUANTIZATION-AWARE TRAINING")
    print("="*80)
    logger.info("Preparing model for Quantization-Aware Training")
    
    # Important: Move model to CPU for QAT preparation
    # QAT requires FP32, not FP16 mixed precision
    model = model.cpu()
    model.train()  # Must be in train mode for QAT
    
    # Prepare model for QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = prepare_qat(model, inplace=True)
    
    # Move back to device
    model = model.to(device)
    
    print("✓ QAT enabled - model will learn quantization-friendly weights")
    print("  - INT8 quantization will be applied after training")
    print("  ⚠️  Mixed precision disabled for QAT (requires FP32)")
    logger.info("QAT preparation complete")
    
    print("="*80 + "\n")
    return model


def convert_to_int8(model, save_path):
    """Convert QAT model to INT8
    
    CRITICAL: Returns the full INT8 model object, not just state_dict!
    The quantized model MUST be used from memory, not reloaded from file.
    """
    print("="*80)
    print("🔄 Converting QAT model to INT8...")
    print("="*80)
    
    # Convert to INT8 (modify in place is safer for quantized models)
    model.eval()  # CRITICAL: Must be in eval mode
    model.cpu()   # INT8 quantization works best on CPU
    
    # Convert the model (inplace=False creates a new model instance)
    print("⚙️  Calling torch.quantization.convert()...")
    model_int8 = convert(model, inplace=False)
    
    # Verify quantization
    quantized_layers = 0
    for name, module in model_int8.named_modules():
        if 'quantized' in str(type(module)).lower() or 'quant' in name.lower():
            quantized_layers += 1
    
    print(f"✓ Quantization verified: {quantized_layers} quantized layers found")
    
    # CRITICAL: Save the FULL model object (not just state_dict!)
    # Quantized models cannot be reconstructed from state_dict alone
    print("💾 Saving full INT8 model object...")
    torch.save(model_int8, save_path)
    
    print(f"✓ INT8 model saved to: {save_path}")
    
    # Check actual size reduction
    import os
    if os.path.exists(save_path):
        int8_size = os.path.getsize(save_path) / (1024*1024)
        print(f"✓ INT8 model size: {int8_size:.2f} MB")
    
    print("\n⚠️  IMPORTANT: Use the returned 'model_int8' variable directly!")
    print("   Do NOT reload from file - quantized models need in-memory object")
    print("="*80 + "\n")
    
    return model_int8
