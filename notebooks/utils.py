import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image


def visualize_embedding_attention(image_path, model, preprocess, device='cuda', figsize=(15, 5)):
    """
    Visualizes which regions of the image are important for the generated embedding.
    Similar to Grad-CAM but using activations from the last convolutional layer.
    """
    # Convert relative path to absolute for Docker
    from pathlib import Path
    if not Path(image_path).is_absolute():
        image_path = str(Path("/workspace") / image_path)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Create model with hooks to capture activations
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # Handle case where output is dict (DeepLabV3) or tensor
            if isinstance(output, dict):
                activation[name] = output
            elif hasattr(output, 'detach'):
                activation[name] = output.detach()
            else:
                activation[name] = output
        return hook

    # Register hook on the last convolutional layer
    # For segmentation models, use the backbone
    if hasattr(model, 'backbone'):
        # DeepLabV3 style - hook on backbone output
        model.backbone.register_forward_hook(get_activation('backbone'))
    elif hasattr(model, 'features'):
        # MobileNetV3 style
        model.features.register_forward_hook(get_activation('features'))
    else:
        # Direct backbone
        model.register_forward_hook(get_activation('backbone'))

    # Ensure model is on the correct device
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        _ = model(img_tensor)

    # Get activations
    if 'backbone' in activation:
        features = activation['backbone']
    elif 'features' in activation:
        features = activation['features']
    else:
        print("No activation maps captured")
        return

    # If features is a dict (like in DeepLab), take the main output
    if isinstance(features, dict):
        # For DeepLab, 'out' is the main output from the backbone
        if 'out' in features:
            features = features['out']
        else:
            # Take the first available output
            features = list(features.values())[0]

    # Create attention map: channel average and upsampling
    attention_map = features.mean(dim=1).squeeze(0)  # [H, W]
    attention_map = torch.relu(attention_map)  # ReLU for positive values

    # Normalize
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Convert to numpy and upsample to original image size
    attention_numpy = attention_map.cpu().numpy()
    attention_resized = cv2.resize(attention_numpy, (img.width, img.height))

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Attention map
    axes[1].imshow(attention_resized, cmap='jet', alpha=0.7)
    axes[1].set_title('Activation Map')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlayed')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return attention_resized


def compare_backbone_attention(image_path, backbone1, backbone2, preprocess,
                              name1="Backbone 1", name2="Backbone 2", device='cuda'):
    """
    Compares attention maps from two different backbones on the same image.
    """
    # Convert relative path to absolute for Docker
    from pathlib import Path
    if not Path(image_path).is_absolute():
        image_path = str(Path("/workspace") / image_path)
    
    img = Image.open(image_path).convert('RGB')

    # Helper function to get attention map
    def get_attention_map(model, img_tensor):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                # Handle case where output is dict (DeepLabV3) or tensor
                if isinstance(output, dict):
                    activation[name] = output
                elif hasattr(output, 'detach'):
                    activation[name] = output.detach()
                else:
                    activation[name] = output
            return hook

        # Register hook
        if hasattr(model, 'backbone'):
            model.backbone.register_forward_hook(get_activation('backbone'))
        elif hasattr(model, 'features'):
            model.features.register_forward_hook(get_activation('features'))
        else:
            model.register_forward_hook(get_activation('model'))

        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            _ = model(img_tensor)

        # Get features
        if 'backbone' in activation:
            features = activation['backbone']
        elif 'features' in activation:
            features = activation['features']
        elif 'model' in activation:
            features = activation['model']
        else:
            return None

        # If features is a dict, take the main output
        if isinstance(features, dict):
            if 'out' in features:
                features = features['out']
            else:
                features = list(features.values())[0]

        # Create attention map
        attention_map = features.mean(dim=1).squeeze(0)
        attention_map = torch.relu(attention_map)
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        return attention_map.cpu().numpy()

    # Preprocess image and ensure correct device
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Ensure models are on the correct device
    backbone1 = backbone1.to(device)
    backbone2 = backbone2.to(device)

    # Get attention maps
    attn1 = get_attention_map(backbone1, img_tensor)
    attn2 = get_attention_map(backbone2, img_tensor)

    if attn1 is None or attn2 is None:
        print("Error getting attention maps")
        return

    # Resize
    attn1_resized = cv2.resize(attn1, (img.width, img.height))
    attn2_resized = cv2.resize(attn2, (img.width, img.height))

    # Calculate difference
    diff = np.abs(attn1_resized - attn2_resized)

    # Visualize comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Backbone 1
    axes[0,0].imshow(img)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')

    axes[0,1].imshow(attn1_resized, cmap='jet')
    axes[0,1].set_title(f'{name1} - Attention')
    axes[0,1].axis('off')

    axes[0,2].imshow(img)
    axes[0,2].imshow(attn1_resized, cmap='jet', alpha=0.5)
    axes[0,2].set_title(f'{name1} - Overlay')
    axes[0,2].axis('off')

    # Empty space
    axes[0,3].axis('off')

    # Row 2: Backbone 2 and comparison
    axes[1,0].imshow(attn2_resized, cmap='jet')
    axes[1,0].set_title(f'{name2} - Activation Map')
    axes[1,0].axis('off')

    axes[1,1].imshow(img)
    axes[1,1].imshow(attn2_resized, cmap='jet', alpha=0.5)
    axes[1,1].set_title(f'{name2} - Overlayed')
    axes[1,1].axis('off')

    axes[1,2].imshow(diff, cmap='hot')
    axes[1,2].set_title('Difference |B1 - B2|')
    axes[1,2].axis('off')

    # Statistics
    axes[1,3].text(0.1, 0.8, f'Similarity: {1 - diff.mean():.3f}', fontsize=12, fontweight='bold')
    axes[1,3].text(0.1, 0.6, f'Mean difference: {diff.mean():.3f}', fontsize=10)
    axes[1,3].text(0.1, 0.4, f'Max difference: {diff.max():.3f}', fontsize=10)
    axes[1,3].axis('off')

    plt.tight_layout()
    plt.show()

    return attn1_resized, attn2_resized, diff


def benchmark_embedding_similarity(embeddings_dict, image_paths_dict, metric='cosine'):
    """
    Calculates similarities between embeddings from different categories/views.
    Useful for evaluating how well the backbone groups views of the same equipment.
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        import seaborn as sns
    except ImportError:
        print("Install scikit-learn and seaborn: pip install scikit-learn seaborn")
        return None

    # Flatten embeddings and create labels
    all_embeddings = []
    labels = []

    for category, emb_list in embeddings_dict.items():
        for i, emb in enumerate(emb_list):
            all_embeddings.append(emb)
            labels.append(f"{category}_{i}")

    all_embeddings = np.array(all_embeddings)

    # Calculate similarity
    if metric == 'cosine':
        similarity_matrix = cosine_similarity(all_embeddings)
    else:
        # Euclidean distance converted to similarity
        dist_matrix = euclidean_distances(all_embeddings)
        similarity_matrix = 1 / (1 + dist_matrix)  # Convert to similarity

    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels,
                cmap='viridis', annot=True, fmt='.2f', square=True)
    plt.title(f'Embedding Similarity Matrix ({metric})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return similarity_matrix


def display_assets_images(assets_path="assets", figsize=(15, 10)):
    assets_path = Path("/workspace") / assets_path

    if not assets_path.exists():
        print(f"Assets path {assets_path} does not exist!")
        return

    subfolders = [f for f in assets_path.iterdir() if f.is_dir()]

    if not subfolders:
        print("No subfolders found in assets directory!")
        return

    print(f"Found {len(subfolders)} categories: {[f.name for f in subfolders]}")

    for subfolder in subfolders:
        category_name = subfolder.name
        image_files = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpeg"))

        if not image_files:
            print(f"No images found in {category_name} folder")
            continue

        print(f"{category_name}: {len(image_files)} images")

        n_images = len(image_files)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        axes = [ax for row in axes for ax in row]

        for i, img_path in enumerate(image_files):
            if i >= len(axes):
                break

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[i].imshow(img_rgb)
            axes[i].set_title(f"{Path(img_path).stem}", fontsize=8)
            axes[i].axis('off')

        for i in range(len(image_files), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Category: {category_name} ({len(image_files)} images)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class MobileNetV3Embeddings:
    def __init__(self, model_size='large', pretrained=True, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)

        if model_size == 'large':
            self.model = models.mobilenet_v3_large(weights='DEFAULT' if pretrained else None)
        elif model_size == 'small':
            self.model = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError("model_size must be 'large' or 'small'")

        self.model.classifier = nn.Identity()
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if isinstance(image, np.ndarray):
            image = self.transform(image)

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image)

        return embedding.squeeze().cpu().numpy()

    def get_embeddings_from_folder(self, folder_path, sample_per_class=1):
        folder_path = Path("/workspace") / folder_path

        if not folder_path.exists():
            raise ValueError(f"Folder {folder_path} does not exist")

        embeddings = {}
        image_paths = {}

        subfolders = [f for f in folder_path.iterdir() if f.is_dir()]

        for subfolder in subfolders:
            category_name = subfolder.name
            image_files = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpeg"))

            if not image_files:
                continue

            sampled_images = np.random.choice(image_files, min(sample_per_class, len(image_files)), replace=False)

            category_embeddings = []
            category_paths = []

            for img_path in sampled_images:
                try:
                    embedding = self.get_embedding(str(img_path))
                    category_embeddings.append(embedding)
                    category_paths.append(str(img_path))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            if category_embeddings:
                embeddings[category_name] = np.array(category_embeddings)
                image_paths[category_name] = category_paths

        return embeddings, image_paths


def plot_embeddings(embeddings, figsize=(12, 8)):
    # Handle both dict and numpy array inputs
    if isinstance(embeddings, dict):
        categories = list(embeddings.keys())
        n_categories = len(categories)

        if n_categories == 0:
            print("No embeddings to plot")
            return

        fig, axes = plt.subplots(n_categories, 1, figsize=figsize)

        if n_categories == 1:
            axes = [axes]

        for i, (category, emb_array) in enumerate(embeddings.items()):
            ax = axes[i]

            if emb_array.ndim == 1:
                emb_array = emb_array.reshape(1, -1)

            n_samples, embedding_dim = emb_array.shape

            im = ax.imshow(emb_array, aspect='auto', cmap='viridis')
            ax.set_title(f'{category} - {n_samples} sample(s), {embedding_dim} dimensions', fontsize=12, fontweight='bold')
            ax.set_xlabel('Embedding Dimension')
            ax.set_ylabel('Sample Index')

            plt.colorbar(im, ax=ax, label='Value')

        plt.tight_layout()
        plt.show()
    else:
        # Handle numpy array input
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_samples, embedding_dim = embeddings.shape

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        im = ax.imshow(embeddings, aspect='auto', cmap='viridis')
        ax.set_title(f'Embeddings - {n_samples} sample(s), {embedding_dim} dimensions', fontsize=12, fontweight='bold')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Sample Index')

        plt.colorbar(im, ax=ax, label='Value')
        plt.tight_layout()
        plt.show()


def print_embedding_stats(embeddings):
    print("=" * 60)
    print("EMBEDDING STATISTICS")
    print("=" * 60)

    # Handle both dict and numpy array inputs
    if isinstance(embeddings, dict):
        for category, emb_array in embeddings.items():
            if emb_array.ndim == 1:
                emb_array = emb_array.reshape(1, -1)

            n_samples, embedding_dim = emb_array.shape

            print(f"\n{category.upper()}:")
            print(f"  Samples: {n_samples}")
            print(f"  Dimensions: {embedding_dim}")
            print(f"  Shape: {emb_array.shape}")
            print(f"  Mean: {emb_array.mean():.4f}")
            print(f"  Std: {emb_array.std():.4f}")
            print(f"  Min: {emb_array.min():.4f}")
            print(f"  Max: {emb_array.max():.4f}")
    else:
        # Handle numpy array input
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_samples, embedding_dim = embeddings.shape

        print(f"\nEMBEDDINGS:")
        print(f"  Samples: {n_samples}")
        print(f"  Dimensions: {embedding_dim}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")

        if n_samples == 1:
            print(f"  First 10 values: {embeddings[0][:10]}")

    print("=" * 60)