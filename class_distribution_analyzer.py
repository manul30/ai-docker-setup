"""
Dataset Class Distribution Analyzer
Helps diagnose class imbalance severity
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def analyze_coco_class_distribution(coco_json_path):
    """
    Analyze class distribution in COCO dataset
    Shows pixel counts, percentages, and imbalance ratios
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Extract class names
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    # Count pixels per class
    class_pixels = {}
    for annotation in coco_data['annotations']:
        cat_id = annotation['category_id']
        area = annotation.get('area', 0)
        
        if cat_id not in class_pixels:
            class_pixels[cat_id] = 0
        class_pixels[cat_id] += area
    
    # Sort by pixel count
    sorted_classes = sorted(class_pixels.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate statistics
    total_pixels = sum(class_pixels.values())
    
    print("\n" + "="*80)
    print(f"CLASS DISTRIBUTION ANALYSIS: {coco_json_path}")
    print("="*80)
    print(f"\nTotal pixels: {total_pixels:,.0f}")
    print(f"Total classes: {len(class_pixels)}")
    
    print(f"\n{'Rank':<6} {'Class Name':<40} {'Pixels':>12} {'Percent':>10} {'Imbalance':>10}")
    print("-"*80)
    
    # Calculate imbalance ratio (first class / each class)
    max_pixels = sorted_classes[0][1]
    
    for rank, (cat_id, pixels) in enumerate(sorted_classes, 1):
        class_name = categories.get(cat_id, f"Unknown_{cat_id}")
        percentage = (pixels / total_pixels) * 100
        imbalance_ratio = max_pixels / pixels if pixels > 0 else float('inf')
        
        # Truncate long names
        if len(class_name) > 38:
            class_name = class_name[:35] + "..."
        
        print(f"{rank:<6} {class_name:<40} {pixels:>12,.0f} {percentage:>9.2f}% {imbalance_ratio:>9.1f}x")
    
    # Print summary statistics
    print("-"*80)
    
    # Calculate imbalance severity
    max_class_pixels = max(class_pixels.values())
    min_class_pixels = min(class_pixels.values())
    
    imbalance_ratio = max_class_pixels / min_class_pixels if min_class_pixels > 0 else float('inf')
    
    print(f"\nImbalance Severity Analysis:")
    print(f"  • Max class pixels: {max_class_pixels:,.0f}")
    print(f"  • Min class pixels: {min_class_pixels:,.0f}")
    print(f"  • Imbalance ratio (max/min): {imbalance_ratio:.1f}x")
    
    # Calculate foreground/background split (assuming class 0 is background)
    if 0 in class_pixels:
        bg_pixels = class_pixels[0]
        fg_pixels = total_pixels - bg_pixels
        bg_percent = (bg_pixels / total_pixels) * 100
        
        print(f"\nForeground/Background Split (assuming class 0 = background):")
        print(f"  • Background: {bg_pixels:,.0f} pixels ({bg_percent:.1f}%)")
        print(f"  • Foreground: {fg_pixels:,.0f} pixels ({100-bg_percent:.1f}%)")
        
        if bg_percent > 80:
            print(f"  ⚠️  SEVERE: Background dominates {bg_percent:.1f}%!")
            print(f"  → Focal Loss highly recommended!")
    
    # Imbalance severity rating
    print(f"\nImbalance Severity Rating:")
    if imbalance_ratio > 100:
        print(f"  🔴 EXTREME: {imbalance_ratio:.1f}x imbalance")
        print(f"  → MUST USE: Focal Loss + Median Frequency Weighting")
    elif imbalance_ratio > 10:
        print(f"  🟠 SEVERE: {imbalance_ratio:.1f}x imbalance")
        print(f"  → Strongly recommended: Focal Loss")
    elif imbalance_ratio > 2:
        print(f"  🟡 MODERATE: {imbalance_ratio:.1f}x imbalance")
        print(f"  → Recommended: Class weighting")
    else:
        print(f"  🟢 MILD: {imbalance_ratio:.1f}x imbalance")
        print(f"  → Standard training may work")
    
    return class_pixels, categories, total_pixels


def visualize_class_distribution(class_pixels, categories, total_pixels):
    """
    Create visualizations of class distribution
    """
    # Sort classes by pixel count
    sorted_classes = sorted(class_pixels.items(), key=lambda x: x[1], reverse=True)
    class_names = [categories.get(cat_id, f"Class_{cat_id}") for cat_id, _ in sorted_classes]
    pixels = [pixels for _, pixels in sorted_classes]
    percentages = [(p / total_pixels) * 100 for p in pixels]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Bar chart of pixel counts
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    ax.barh(range(len(class_names)), pixels, color=colors)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels([name[:30] for name in class_names], fontsize=9)
    ax.set_xlabel('Pixels', fontsize=11, fontweight='bold')
    ax.set_title('Pixel Count per Class', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: Pie chart of distribution
    ax = axes[0, 1]
    ax.pie(percentages, labels=[name[:20] for name in class_names], autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 8})
    ax.set_title('Class Distribution (%)', fontsize=12, fontweight='bold')
    
    # Plot 3: Log scale bar chart (to see rare classes)
    ax = axes[1, 0]
    ax.barh(range(len(class_names)), pixels, color=colors)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels([name[:30] for name in class_names], fontsize=9)
    ax.set_xlabel('Pixels (log scale)', fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_title('Pixel Count per Class (Log Scale)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 4: Cumulative distribution
    ax = axes[1, 1]
    cumsum_percent = np.cumsum(percentages)
    ax.plot(range(len(class_names)), cumsum_percent, marker='o', linewidth=2, markersize=6)
    ax.axhline(y=80, color='r', linestyle='--', label='80%', linewidth=2)
    ax.axhline(y=90, color='orange', linestyle='--', label='90%', linewidth=2)
    ax.set_xlabel('Class Rank', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative %', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('class_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: class_distribution_analysis.png")
    plt.show()


if __name__ == "__main__":
    # Example usage
    dataset_path = Path("./hvac-segmentation-automatic-flow-8/train/_annotations.coco.json")
    
    if dataset_path.exists():
        class_pixels, categories, total_pixels = analyze_coco_class_distribution(str(dataset_path))
        visualize_class_distribution(class_pixels, categories, total_pixels)
    else:
        print(f"Error: Dataset not found at {dataset_path}")
        print("\nUsage:")
        print("  python -c 'from class_distribution_analyzer import *'")
        print("  analyze_coco_class_distribution('path/to/_annotations.coco.json')")
