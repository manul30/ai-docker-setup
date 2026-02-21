"""
Dataset utilities for COCO object detection
"""
import os
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import CocoDetection
from PIL import Image


class YOLOStyleAugmentation:
    """YOLO-inspired aggressive augmentation for object detection"""
    
    def __init__(self, img_size=320, augment_prob=0.8):
        self.img_size = img_size
        self.augment_prob = augment_prob
        
    def __call__(self, img, target):
        """Apply random augmentations"""
        boxes = target['boxes']
        labels = target['labels']
        
        # Random horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            # Flip boxes
            w = img.width
            boxes_np = boxes.numpy().copy()
            boxes_np[:, [0, 2]] = w - boxes_np[:, [2, 0]]
            boxes = torch.from_numpy(boxes_np)
        
        # Color jitter (YOLO HSV augmentation)
        if random.random() < self.augment_prob:
            img = T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.7,
                hue=0.015
            )(img)
        
        # Random blur
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            img = TF.gaussian_blur(img, kernel_size)
        
        # Random grayscale
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        
        # Convert to tensor and normalize
        img = TF.to_tensor(img)
        
        # Random noise
        if random.random() < 0.2:
            noise = torch.randn_like(img) * 0.05
            img = torch.clamp(img + noise, 0, 1)
        
        target['boxes'] = boxes
        target['labels'] = labels
        
        return img, target


class COCODataset(CocoDetection):
    """COCO dataset for object detection with augmentation support"""
    
    def __init__(self, root, annFile, transforms=None, img_size=320, use_augmentation=False):
        super().__init__(root, annFile)
        self.transforms = transforms
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        
        if use_augmentation:
            self.augmentation = YOLOStyleAugmentation(img_size=img_size, augment_prob=0.8)
            print("✓ YOLO-style augmentation ENABLED")
        
        # Filter out images without annotations
        self.valid_ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.valid_ids.append(img_id)
        
        print(f"Filtered dataset: {len(self.valid_ids)}/{len(self.ids)} images have annotations")
        
    def __len__(self):
        return len(self.valid_ids)
        
    def __getitem__(self, idx):
        # Use valid_ids instead of ids
        img_id = self.valid_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        # Parse annotations BEFORE resizing for proper coordinate scaling
        boxes = []
        labels = []
        img_w, img_h = img.size
        
        for obj in anns:
            bbox = obj['bbox']  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            # Skip invalid boxes
            if w > 0 and h > 0:
                boxes.append([x1, y1, x2, y2])
                labels.append(obj['category_id'])
        
        # Ensure we have at least one box
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [1]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Resize image and scale boxes
        scale_x = self.img_size / img_w
        scale_y = self.img_size / img_h
        img = TF.resize(img, [self.img_size, self.img_size])
        
        # Scale boxes to new image size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        target_dict = {
            'boxes': boxes,
            'labels': labels,
        }
        
        # Apply augmentation if enabled
        if self.use_augmentation:
            img, target_dict = self.augmentation(img, target_dict)
        else:
            img = TF.to_tensor(img)
        
        return img, target_dict
    
    def get_class_name(self):
        return self.coco.cats[1]['name']


def collate_fn(batch):
    """Collate function for DataLoader"""
    return tuple(zip(*batch))
