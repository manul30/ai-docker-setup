"""
Model export functions for mobile deployment.

Key design: SSDExportWrapper bypasses ssd.transform() and ssd.postprocess()
so torch.jit.trace never touches pybind11 ops (which hang the tracer).
Input contract: [1, 3, H, W] float32, ImageNet-normalised by the caller.
"""
import os
import subprocess
import sys

import torch
import torch.nn as nn
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType


# ── shared wrapper (used by both ONNX and CoreML exports) ──────────────────

class SSDExportWrapper(nn.Module):
    """Trace-safe SSD wrapper.

    Skips ssd.transform() and ssd.postprocess() — both contain pybind11
    ops that cause torch.jit.trace to hang indefinitely.

    Input:  [1, 3, H, W] float32, ImageNet-normalised by the caller.
    Output: (bbox_regression, cls_logits) — raw head tensors.
    """
    def __init__(self, ssd_model):
        super().__init__()
        self.backbone = ssd_model.backbone
        self.head     = ssd_model.head

    def forward(self, x):
        features  = self.backbone(x)
        feat_list = list(features.values()) if isinstance(features, dict) else [features]
        out       = self.head(feat_list)
        return out["bbox_regression"], out["cls_logits"]


def _dummy_input(config):
    """Normalised dummy input (ImageNet mean/std)."""
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (torch.rand(1, 3, config.IMG_SIZE, config.IMG_SIZE) - _mean) / _std


# ── ONNX (Android) ─────────────────────────────────────────────────────────

def export_to_onnx(model, config):
    """Export to ONNX FP32 + dynamic-INT8.

    Produces:
      - detection_mobilenetv3_fp32.onnx
      - detection_mobilenetv3_int8.onnx  (onnxruntime dynamic quantisation)
    """
    print("=" * 80)
    print("🤖 ANDROID EXPORT — ONNX")
    print("=" * 80 + "\n")

    DATA     = "/workspace/data"
    fp32_path = os.path.join(DATA, "detection_mobilenetv3_fp32.onnx")
    int8_path = os.path.join(DATA, "detection_mobilenetv3_int8.onnx")

    model.eval().cpu()
    wrapper = SSDExportWrapper(model).eval()
    dummy   = _dummy_input(config)

    # ── FP32 export ────────────────────────────────────────────────────────
    print("Exporting FP32 ONNX…")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            fp32_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["bbox_regression", "cls_logits"],
        )

    # Verify
    onnx.checker.check_model(fp32_path)
    mb = os.path.getsize(fp32_path) / 1024 / 1024
    print(f"  ✅ FP32 → {fp32_path}  ({mb:.1f} MB)")

    # Quick ORT inference check
    sess = onnxruntime.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    _ = sess.run(None, {"input": dummy.numpy()})
    print("  ✓ ORT inference verified")

    # ── INT8 export (dynamic quantisation) ────────────────────────────────
    print("\nQuantising to INT8 (dynamic)…")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    mb8 = os.path.getsize(int8_path) / 1024 / 1024
    print(f"  ✅ INT8 → {int8_path}  ({mb8:.1f} MB)")

    sess8 = onnxruntime.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    _ = sess8.run(None, {"input": dummy.numpy()})
    print("  ✓ ORT INT8 inference verified")

    print("\n" + "=" * 80)
    return fp32_path, int8_path


# ── CoreML (iOS) ───────────────────────────────────────────────────────────

def export_to_ios(model, config):
    """Export to CoreML FP32 + INT8 (linear weight quantisation).

    Produces:
      - detection_mobilenetv3_fp32.mlpackage
      - detection_mobilenetv3_int8.mlpackage
    """
    print("=" * 80)
    print("🍎 iOS EXPORT — CoreML")
    print("=" * 80 + "\n")

    # Auto-install coremltools if missing
    try:
        import coremltools as ct
        print(f"✓ coremltools {ct.__version__}")
    except ImportError:
        print("  Installing coremltools…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "coremltools>=7.0"])
        import coremltools as ct

    DATA      = "/workspace/data"
    fp32_path = os.path.join(DATA, "detection_mobilenetv3_fp32.mlpackage")
    int8_path = os.path.join(DATA, "detection_mobilenetv3_int8.mlpackage")

    model.eval().cpu()
    wrapper = SSDExportWrapper(model).eval()
    dummy   = _dummy_input(config)

    def _convert(quantize_weights=False):
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, dummy, strict=False)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=dummy.shape)],
            minimum_deployment_target=ct.target.iOS15,
            convert_to="mlprogram",
        )

        if quantize_weights:
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_channel",
            )
            cfg = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, cfg)

        return mlmodel

    # FP32
    print("Converting FP32…")
    ml_fp32 = _convert(quantize_weights=False)
    ml_fp32.save(fp32_path)
    mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(fp32_path) for f in fns
    ) / 1024 / 1024
    print(f"  ✅ FP32 → {fp32_path}  ({mb:.1f} MB)")

    # INT8
    print("\nConverting INT8 (linear weight quantisation)…")
    ml_int8 = _convert(quantize_weights=True)
    ml_int8.save(int8_path)
    mb8 = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(int8_path) for f in fns
    ) / 1024 / 1024
    print(f"  ✅ INT8 → {int8_path}  ({mb8:.1f} MB)")

    print("\n" + "=" * 80)
    print("💡 Drag the .mlpackage into Xcode → Vision framework → VNCoreMLRequest")
    print("=" * 80)
    return fp32_path, int8_path


# ── Android TorchScript (legacy, kept for reference) ───────────────────────

def export_to_android(model, config):
    """Export to TorchScript Mobile (.ptl).

    Note: prefer export_to_onnx() for Android — ONNX Runtime is smaller
    and avoids the PyTorch Mobile dependency.
    """
    from torch.utils.mobile_optimizer import optimize_for_mobile

    print("=" * 80)
    print("🤖 ANDROID EXPORT — TorchScript Mobile (.ptl)")
    print("=" * 80 + "\n")

    model.eval().cpu()
    wrapper = SSDExportWrapper(model).eval()
    dummy   = _dummy_input(config)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy, strict=False)

    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(config.ANDROID_MODEL_PATH)

    mb = os.path.getsize(config.ANDROID_MODEL_PATH) / 1024 / 1024
    print(f"  ✅ TorchScript Mobile → {config.ANDROID_MODEL_PATH}  ({mb:.1f} MB)")
    print("\n" + "=" * 80)
    return config.ANDROID_MODEL_PATH
