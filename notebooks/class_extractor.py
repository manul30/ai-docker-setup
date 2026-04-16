"""
Automatic COCO Dataset Class and Color Extractor

This utility automatically:
1. Reads classes from _annotations.coco.json
2. Generates distinct colors for each class
3. Provides visualization-ready data structures
"""

import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


def extract_classes_from_coco(coco_json_path: str) -> Dict[int, str]:
    """
    Extract class names from COCO annotations JSON file.
    
    Args:
        coco_json_path: Path to _annotations.coco.json file
        
    Returns:
        Dictionary mapping category_id -> category_name
        
    Example:
        >>> classes = extract_classes_from_coco("dataset/train/_annotations.coco.json")
        >>> print(classes)
        {0: 'background', 1: 'HVAC', 2: 'compressor', ...}
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    classes = {}
    if 'categories' in coco_data:
        for category in coco_data['categories']:
            cat_id = category['id']
            cat_name = category['name']
            classes[cat_id] = cat_name
    
    return classes


def generate_distinct_colors(num_colors: int, seed: int = 42) -> List[List[int]]:
    """
    Generate visually distinct colors using HSV color space.
    
    Strategy:
    - Distribute colors evenly in HSV space
    - High saturation and value for visibility
    - Convert to RGB
    
    Args:
        num_colors: Number of distinct colors needed
        seed: Random seed for reproducibility
        
    Returns:
        List of RGB colors, each [R, G, B] with values 0-255
        
    Example:
        >>> colors = generate_distinct_colors(10)
        >>> print(len(colors))
        10
        >>> print(colors[0])
        [255, 0, 0]  # Red
    """
    np.random.seed(seed)
    
    # Create colors in HSV space for better distribution
    colors_hsv = []
    
    # Special cases for first few classes
    special_colors = [
        (0, 0, 0),           # 0: Black (background)
        (0, 255, 255),       # 1: Red
        (120, 255, 255),     # 2: Green
        (240, 255, 255),     # 3: Blue
        (30, 255, 255),      # 4: Yellow
        (300, 255, 255),     # 5: Magenta
        (180, 255, 255),     # 6: Cyan
    ]
    
    # Add special colors
    for i in range(min(len(special_colors), num_colors)):
        colors_hsv.append(special_colors[i])
    
    # Generate remaining colors by distributing evenly in hue space
    remaining = num_colors - len(colors_hsv)
    if remaining > 0:
        hues = np.linspace(0, 360, remaining, endpoint=False)
        for hue in hues:
            saturation = 200 + np.random.randint(-50, 50)  # 150-250
            value = 200 + np.random.randint(-30, 30)        # 170-230
            colors_hsv.append((hue, saturation, value))
    
    # Convert HSV to RGB
    colors_rgb = []
    for h, s, v in colors_hsv:
        # HSV to RGB conversion
        h_i = int(h / 60) % 6
        f = (h / 60) - int(h / 60)
        
        p = v * (1 - s / 255)
        q = v * (1 - f * s / 255)
        t = v * (1 - (1 - f) * s / 255)
        
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        colors_rgb.append([int(r), int(g), int(b)])
    
    return colors_rgb


def load_coco_classes_and_colors(
    dataset_path: str,
    split: str = 'train',
    auto_generate_colors: bool = True,
    manual_colors: List[List[int]] = None,
    seed: int = 42
) -> Tuple[Dict[int, str], List[List[int]], int]:
    """
    Load classes and generate/provide colors from COCO dataset.
    
    Args:
        dataset_path: Path to dataset root directory
        split: Dataset split ('train', 'valid', 'test')
        auto_generate_colors: If True, generate colors automatically
        manual_colors: If provided, use these colors instead of generating
        seed: Random seed for color generation
        
    Returns:
        Tuple of:
        - classes_dict: Dict mapping category_id -> category_name
        - colors_list: List of [R, G, B] colors for each class
        - num_classes: Total number of classes
        
    Example:
        >>> classes, colors, num_classes = load_coco_classes_and_colors(
        ...     "dataset/hvac-segmentation",
        ...     split='train'
        ... )
        >>> print(f"Classes: {classes}")
        >>> print(f"Colors: {colors}")
        >>> print(f"Num classes: {num_classes}")
    """
    dataset_path = Path(dataset_path)
    coco_json = dataset_path / split / "_annotations.coco.json"
    
    if not coco_json.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_json}")
    
    # Extract classes
    classes = extract_classes_from_coco(str(coco_json))
    
    # Determine number of classes
    # The max category ID + 1, or explicit num_classes key if present
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    if 'info' in coco_data and 'num_classes' in coco_data['info']:
        num_classes = coco_data['info']['num_classes']
    else:
        # Infer from max category ID
        max_cat_id = max(classes.keys()) if classes else 0
        num_classes = max_cat_id + 1
    
    # Generate or use provided colors
    if manual_colors is not None:
        colors = manual_colors
    elif auto_generate_colors:
        colors = generate_distinct_colors(num_classes, seed=seed)
    else:
        # Fallback to simple color palette
        colors = generate_distinct_colors(num_classes, seed=seed)
    
    # Create ordered list indexed by category ID
    class_names = []
    for i in range(num_classes):
        if i in classes:
            class_names.append(classes[i])
        else:
            class_names.append(f"Class_{i}")
    
    return classes, colors, num_classes, class_names


def visualize_class_colors(classes_dict: Dict[int, str], colors: List[List[int]]):
    """
    Display a color palette of all classes.
    
    Args:
        classes_dict: Mapping of category_id -> category_name
        colors: List of RGB colors
    """
    import matplotlib.pyplot as plt
    
    num_classes = len(classes_dict)
    
    # Create figure with color swatches
    fig, ax = plt.subplots(figsize=(12, max(6, num_classes * 0.3)))
    
    for idx, (cat_id, cat_name) in enumerate(sorted(classes_dict.items())):
        # Get color
        color_rgb = colors[cat_id]
        color_normalized = [c / 255.0 for c in color_rgb]
        
        # Draw color swatch
        ax.add_patch(plt.Rectangle((0, idx), 1, 0.8, facecolor=color_normalized, edgecolor='black'))
        
        # Add text label
        ax.text(1.2, idx + 0.4, f"[{cat_id}] {cat_name}", va='center', fontsize=11, fontweight='bold')
        ax.text(1.2, idx + 0.1, f"RGB{tuple(color_rgb)}", va='center', fontsize=9, family='monospace', color='gray')
    
    ax.set_xlim(-0.1, 4)
    ax.set_ylim(-0.5, num_classes)
    ax.axis('off')
    
    plt.title("Dataset Class Color Palette", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("COCO Class Extractor - Example Usage")
    print("=" * 70)
    
    # Example: extract from dataset
    dataset_path = "./hvac-segmentation-automatic-flow-8"
    
    try:
        classes, colors, num_classes, class_names = load_coco_classes_and_colors(
            dataset_path,
            split='train',
            auto_generate_colors=True
        )
        
        print(f"\nExtracted {num_classes} classes:")
        for idx, name in enumerate(class_names):
            color = colors[idx]
            print(f"  [{idx:2d}] {name:30s} RGB{tuple(color)}")
        
        print("\nClasses dictionary (category_id -> name):")
        for cat_id, name in sorted(classes.items()):
            print(f"  {cat_id}: {name}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have a COCO dataset at the specified path.")
