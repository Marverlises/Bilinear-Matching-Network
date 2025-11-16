"""
Comprehensive visualization script for BMNet/BMNet+
Generates all visualizations from the paper and additional useful visualizations
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from config import cfg
from FSC147_dataset import build_dataset, batch_collate_fn
from models import build_model
from util.visualization import Visualizer
import util.misc as utils


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def get_exemplar_boxes(annotation_file, file_name, scale_factor=1.0):
    """Get exemplar boxes from annotation file"""
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    if file_name not in annotations:
        return None
    
    boxes = np.array(annotations[file_name]['box_examples_coordinates'])
    boxes = boxes * scale_factor
    return boxes[:3]  # Return first 3 exemplars


def main(args):
    print("=" * 80)
    print("BMNet/BMNet+ Comprehensive Visualization Tool")
    print("=" * 80)
    
    # Setup device
    device = torch.device(cfg.TRAIN.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Build dataset
    print(f"\nBuilding dataset from: {cfg.DIR.dataset}")
    dataset = build_dataset(cfg, is_train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.TRAIN.num_workers,
        collate_fn=batch_collate_fn
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create visualizer
    output_dir = args.output_dir or os.path.join(cfg.DIR.result, 'visualizations')
    visualizer = Visualizer(output_dir, cmap=args.cmap)
    print(f"\nVisualizations will be saved to: {output_dir}")
    
    # Load annotations for exemplar boxes
    annotation_file = os.path.join(cfg.DIR.dataset, 'annotation_FSC147_384.json')
    has_annotations = os.path.exists(annotation_file)
    
    # Process samples
    num_samples = args.num_samples if args.num_samples > 0 else len(data_loader)
    print(f"\nProcessing {num_samples} samples...")
    print("-" * 80)
    
    mae_sum = 0
    mse_sum = 0
    
    for idx, sample in enumerate(data_loader):
        if idx >= num_samples:
            break
        
        img, patches, targets = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)
        
        # Get file name
        file_name = dataset.data_list[idx][0]
        print(f"\n[{idx+1}/{num_samples}] Processing: {file_name}")
        
        # Load original image
        image_path = os.path.join(cfg.DIR.dataset, 'images_384_VarV2', file_name)
        if not os.path.exists(image_path):
            print(f"  Warning: Image not found at {image_path}, skipping...")
            continue
        
        original_img = Image.open(image_path).convert("RGB")
        original_img = np.array(original_img)
        
        # Calculate scale factor
        h_orig, w_orig = original_img.shape[:2]
        h_model, w_model = img.shape[-2:]
        scale_factor = min(h_orig / h_model, w_orig / w_model)
        
        # Get exemplar boxes
        exemplar_boxes = None
        if has_annotations:
            exemplar_boxes = get_exemplar_boxes(annotation_file, file_name, scale_factor)
        
        # Run model with intermediate outputs
        with torch.no_grad():
            outputs = model(img, patches, is_train=False, return_intermediate=True)
        
        # Get counts
        pred_count = outputs['density_map'].sum().item()
        gt_count = targets['gtcount'].item()
        
        # Calculate metrics
        error = abs(pred_count - gt_count)
        mae_sum += error
        mse_sum += error ** 2
        
        print(f"  GT Count: {gt_count:.1f}, Pred Count: {pred_count:.1f}, Error: {error:.2f}")
        
        # Process correlation map - convert from [B, H*W, num_exemplars] to [B, H, W]
        corr_map = outputs['corr_map']
        if isinstance(corr_map, torch.Tensor):
            if corr_map.dim() == 3:  # [B, H*W, num_exemplars] or [B, H, W]
                bs, first_dim, second_dim = corr_map.shape
                if first_dim > second_dim:  # [B, H*W, num_exemplars]
                    # Need to reshape to [B, H, W, num_exemplars] then average
                    # We need to know H and W from the feature map
                    h, w = outputs['refined_features'].shape[-2:]
                    corr_map = corr_map.view(bs, h, w, second_dim).mean(dim=-1)  # [B, H, W]
                # else it's already [B, H, W]
        
        # Prepare outputs for visualization
        vis_outputs = {
            'density_map': outputs['density_map'],
            'corr_map': corr_map,
            'attention_maps': outputs.get('attention_maps'),
            'dynamic_weights': outputs.get('dynamic_weights'),
            'refined_features': outputs.get('refined_features'),
            'patch_features': outputs.get('patch_features'),
            'query_features': outputs.get('query_features')
        }
        
        # Get point map and ground truth density map
        pt_map = targets['pt_map'].cpu().numpy()[0, 0] if 'pt_map' in targets else None
        gt_density = targets['density_map'].cpu().numpy()[0, 0] if 'density_map' in targets else None
        
        # Create comprehensive visualization
        base_name = os.path.splitext(file_name)[0]
        visualizer.create_comprehensive_visualization(
            vis_outputs,
            original_img,
            exemplar_boxes=exemplar_boxes,
            pt_map=pt_map,
            gt_density=gt_density,
            gt_count=gt_count,
            pred_count=pred_count,
            file_name=base_name,
            save_dir=output_dir
        )
        
        print(f"  ✓ Saved visualizations for {base_name}")
    
    # Print summary
    mae = mae_sum / num_samples
    mse = (mse_sum / num_samples) ** 0.5
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Processed: {num_samples} samples")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  Visualizations saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive Visualization for BMNet/BMNet+')
    parser.add_argument('--cfg', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations (default: ./results/visualizations)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize (default: 10, use -1 for all)')
    parser.add_argument('--cmap', type=str, default='jet',
                       help='Colormap for heatmaps (default: jet)')
    
    args = parser.parse_args()
    
    # Load config
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    
    main(args)

