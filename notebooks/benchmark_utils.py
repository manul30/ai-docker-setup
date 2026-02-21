"""
Benchmarking utilities for model evaluation
"""
import torch
import time
import numpy as np
import os
from thop import profile, clever_format


def benchmark_model_size(fp32_path, int8_path, config, get_model_fn):
    """Compare model sizes"""
    print("="*80)
    print("📏 MODEL SIZE COMPARISON")
    print("="*80 + "\n")
    
    def get_model_size(path):
        if os.path.exists(path):
            size_bytes = os.path.getsize(path)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        return None
    
    # Load FP32 model for parameter counting
    model_fp32 = get_model_fn(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(fp32_path)
    model_fp32.load_state_dict(checkpoint['model_state_dict'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model_fp32.parameters())
    trainable_params = sum(p.numel() for p in model_fp32.parameters() if p.requires_grad)
    
    print(f"📦 FP32 Model (Full Precision + Pruning):")
    print(f"   File size: {get_model_size(fp32_path):.2f} MB")
    print(f"   Parameters: {total_params:,}")
    print(f"   Trainable:  {trainable_params:,}")
    
    # Calculate theoretical FLOPs
    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    flops, params = profile(model_fp32, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"   FLOPs: {flops}")
    print(f"   Params (THOP): {params}\n")
    
    # INT8 model info
    if os.path.exists(int8_path):
        int8_size = get_model_size(int8_path)
        print(f"🎯 INT8 Model (Quantized):")
        print(f"   File size: {int8_size:.2f} MB")
        print(f"   Size reduction: {(1 - int8_size/get_model_size(fp32_path))*100:.1f}%")
        print(f"   Parameters: {total_params:,} (same count, 4x less memory)")
        print(f"   Expected speedup: 2-4x on CPU\n")
    
    print("="*80)


def benchmark_inference_speed(model_fp32, model_int8, config, device):
    """Benchmark inference speed on GPU and CPU"""
    print("="*80)
    print("⚡ INFERENCE SPEED BENCHMARK")
    print("="*80 + "\n")
    
    # Prepare test input
    test_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    warmup_runs = 10
    benchmark_runs = 100
    
    def benchmark_model(model, input_tensor, device, warmup=10, runs=100):
        """Benchmark model inference time"""
        model.eval()
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                _ = model(input_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    # Benchmark on GPU
    print("🖥️  GPU Benchmark (CUDA):")
    if torch.cuda.is_available():
        # FP32 Model
        model_fp32.eval()
        fp32_gpu_stats = benchmark_model(model_fp32, test_input, device, warmup_runs, benchmark_runs)
        print(f"\n   FP32 (Pruned):")
        print(f"     Mean: {fp32_gpu_stats['mean']:.2f} ms ± {fp32_gpu_stats['std']:.2f} ms")
        print(f"     Median: {fp32_gpu_stats['median']:.2f} ms")
        print(f"     Min/Max: {fp32_gpu_stats['min']:.2f} / {fp32_gpu_stats['max']:.2f} ms")
        
        # INT8 Model (if available)
        if model_int8:
            int8_gpu_stats = benchmark_model(model_int8, test_input, device, warmup_runs, benchmark_runs)
            print(f"\n   INT8 (Quantized):")
            print(f"     Mean: {int8_gpu_stats['mean']:.2f} ms ± {int8_gpu_stats['std']:.2f} ms")
            print(f"     Median: {int8_gpu_stats['median']:.2f} ms")
            print(f"     Min/Max: {int8_gpu_stats['min']:.2f} / {int8_gpu_stats['max']:.2f} ms")
            print(f"     Speedup: {fp32_gpu_stats['mean'] / int8_gpu_stats['mean']:.2f}x")
    else:
        print("   CUDA not available")
    
    # Benchmark on CPU
    print("\n\n💻 CPU Benchmark:")
    cpu_device = torch.device('cpu')
    
    # FP32 Model
    fp32_cpu_stats = benchmark_model(model_fp32, test_input, cpu_device, warmup_runs, benchmark_runs)
    print(f"\n   FP32 (Pruned):")
    print(f"     Mean: {fp32_cpu_stats['mean']:.2f} ms ± {fp32_cpu_stats['std']:.2f} ms")
    print(f"     Median: {fp32_cpu_stats['median']:.2f} ms")
    print(f"     Min/Max: {fp32_cpu_stats['min']:.2f} / {fp32_cpu_stats['max']:.2f} ms")
    
    # INT8 Model (if available)
    if model_int8:
        int8_cpu_stats = benchmark_model(model_int8, test_input, cpu_device, warmup_runs, benchmark_runs)
        print(f"\n   INT8 (Quantized):")
        print(f"     Mean: {int8_cpu_stats['mean']:.2f} ms ± {int8_cpu_stats['std']:.2f} ms")
        print(f"     Median: {int8_cpu_stats['median']:.2f} ms")
        print(f"     Min/Max: {int8_cpu_stats['min']:.2f} / {int8_cpu_stats['max']:.2f} ms")
        print(f"     Speedup: {fp32_cpu_stats['mean'] / int8_cpu_stats['mean']:.2f}x")
        
        # Check if meets target
        target_latency = 20  # ms
        if int8_cpu_stats['mean'] < target_latency:
            print(f"\n   ✓ Target latency achieved: {int8_cpu_stats['mean']:.2f} ms < {target_latency} ms")
        else:
            print(f"\n   ⚠ Target latency not met: {int8_cpu_stats['mean']:.2f} ms > {target_latency} ms")
    
    print("\n" + "="*80)
