"""
Image processing utility functions - Process single image for web application
"""
import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from FSC147_dataset import pad_to_constant, MainTransform, get_query_transforms


def process_image_for_inference(image_path, exemplar_boxes=None, min_size=384, max_size=1584, 
                                 box_number=3, scale_number=20, exemplar_size=(128, 128)):
    """
    Process single image for model inference
    
    Args:
        image_path: Image file path or PIL Image object
        exemplar_boxes: List of exemplar boxes, format [[x1, y1, x2, y2], ...] or None (auto-select)
        min_size: Minimum image size
        max_size: Maximum image size
        box_number: Number of exemplar boxes
        scale_number: Number of scale embeddings
        exemplar_size: Resized exemplar box size
    
    Returns:
        img_tensor: Processed image tensor [1, C, H, W]
        patches_dict: Dictionary containing patches and scale_embedding
        original_img: Original image array [H, W, 3]
        scale_factor: Scale factor
    """
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")
    
    w, h = img.size
    original_img = np.array(img)
    
    # Resize image
    r = 1.0
    if h > max_size or w > max_size:
        r = max_size / max(h, w)
    if r * h < min_size or w * r < min_size:
        r = min_size / min(h, w)
    nh, nw = int(r * h), int(r * w)
    img = img.resize((nw, nh), resample=Image.BICUBIC)
    
    # Exemplar boxes are required (user draws them in frontend)
    if exemplar_boxes is None or len(exemplar_boxes) == 0:
        raise ValueError("Exemplar boxes are required. Please draw at least 1 exemplar box on the image.")
    
    # Adjust exemplar box coordinates by scale factor (from original image size to resized size)
    exemplar_boxes = (np.array(exemplar_boxes) * r).astype(np.int32).tolist()
    exemplar_boxes = exemplar_boxes[:box_number]  # Take at most first 3
    
    # Crop patches and calculate scale embedding
    query_transform = get_query_transforms(is_train=False, exemplar_size=exemplar_size)
    patches = []
    scale_embedding = []
    
    for box in exemplar_boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, nw - 1))
        y1 = max(0, min(y1, nh - 1))
        x2 = max(x1 + 1, min(x2, nw))
        y2 = max(y1 + 1, min(y2, nh))
        
        patch = img.crop((x1, y1, x2, y2))
        patches.append(query_transform(patch))
        
        # Calculate scale embedding
        scale = (x2 - x1) / nw * 0.5 + (y2 - y1) / nh * 0.5
        scale = scale // (0.5 / scale_number)
        scale = int(scale) if scale < scale_number - 1 else scale_number - 1
        scale_embedding.append(scale)
    
    # Apply main transform
    main_transform = MainTransform()
    target = {
        'density_map': np.zeros((nh, nw), dtype=np.float32),
        'pt_map': np.zeros((nh, nw), dtype=np.int32),
        'gtcount': 0
    }
    img_tensor, target = main_transform(img, target)
    
    # Stack patches
    patches_tensor = torch.stack(patches, dim=0)
    scale_embedding_tensor = torch.tensor(scale_embedding, dtype=torch.long)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    patches_tensor = patches_tensor.unsqueeze(0)  # [1, num_patches, C, H, W]
    scale_embedding_tensor = scale_embedding_tensor.unsqueeze(0)  # [1, num_patches]
    
    patches_dict = {
        'patches': patches_tensor,
        'scale_embedding': scale_embedding_tensor
    }
    
    scale_factor = r
    
    return img_tensor, patches_dict, original_img, scale_factor


def auto_select_exemplar_boxes(width, height, num_boxes=3):
    """
    Automatically select exemplar boxes (evenly distributed from image center region)
    
    Args:
        width: Image width
        height: Image height
        num_boxes: Number of boxes needed
    
    Returns:
        boxes: List of exemplar boxes [[x1, y1, x2, y2], ...]
    """
    boxes = []
    center_x, center_y = width // 2, height // 2
    
    # Calculate box size (approximately 1/10 of image)
    box_size = min(width, height) // 10
    box_size = max(32, min(box_size, 128))  # Limit between 32-128
    
    if num_boxes == 1:
        # Single box at center
        x1 = max(0, center_x - box_size // 2)
        y1 = max(0, center_y - box_size // 2)
        x2 = min(width, x1 + box_size)
        y2 = min(height, y1 + box_size)
        boxes.append([x1, y1, x2, y2])
    elif num_boxes == 2:
        # Two boxes, left-right distribution
        offset = box_size
        for dx in [-offset, offset]:
            x1 = max(0, center_x + dx - box_size // 2)
            y1 = max(0, center_y - box_size // 2)
            x2 = min(width, x1 + box_size)
            y2 = min(height, y1 + box_size)
            boxes.append([x1, y1, x2, y2])
    else:
        # Three or more boxes, evenly distributed
        if num_boxes == 3:
            # Triangular distribution
            positions = [
                (center_x, center_y - box_size),  # Top
                (center_x - box_size, center_y + box_size // 2),  # Bottom left
                (center_x + box_size, center_y + box_size // 2),  # Bottom right
            ]
        else:
            # Grid distribution
            grid_size = int(np.ceil(np.sqrt(num_boxes)))
            step_x = width // (grid_size + 1)
            step_y = height // (grid_size + 1)
            positions = []
            for i in range(num_boxes):
                row = i // grid_size
                col = i % grid_size
                x = step_x * (col + 1)
                y = step_y * (row + 1)
                positions.append((x, y))
        
        for px, py in positions[:num_boxes]:
            x1 = max(0, px - box_size // 2)
            y1 = max(0, py - box_size // 2)
            x2 = min(width, x1 + box_size)
            y2 = min(height, y1 + box_size)
            boxes.append([x1, y1, x2, y2])
    
    return boxes

