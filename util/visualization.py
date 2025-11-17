"""
Comprehensive visualization tools for BMNet/BMNet+ model
Includes all visualizations from the paper and additional useful visualizations
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

plt.switch_backend('agg')
sns.set_style("whitegrid")


class Visualizer:
    """Comprehensive visualizer for BMNet/BMNet+ model"""

    def __init__(self, output_dir, cmap='jet'):
        """
        Args:
            output_dir: Directory to save visualizations
            cmap: Colormap for heatmaps
        """
        self.output_dir = output_dir
        self.cmap = plt.cm.get_cmap(cmap)
        os.makedirs(output_dir, exist_ok=True)

    def visualize_correlation_map(self, corr_map, original_img, exemplar_boxes=None,
                                  save_path=None, title="Correlation Map"):
        """
        Visualize correlation/similarity map (Fig. 1 in paper)

        Args:
            corr_map: Similarity map tensor [B, H, W] or [H, W]
            original_img: Original image array [H, W, 3]
            exemplar_boxes: List of exemplar boxes [[x1, y1, x2, y2], ...]
            save_path: Path to save the visualization
            title: Title for the figure
        """
        if isinstance(corr_map, torch.Tensor):
            if corr_map.dim() == 4:  # [B, 1, H, W] or [B, H, W]
                corr_map = corr_map[0]
            if corr_map.dim() == 3:
                corr_map = corr_map[0]
            corr_map = corr_map.cpu().numpy()

        # Resize correlation map to match original image size
        h_orig, w_orig = original_img.shape[:2]
        h_corr, w_corr = corr_map.shape

        if h_corr != h_orig or w_corr != w_orig:
            corr_map = F.interpolate(
                torch.from_numpy(corr_map).unsqueeze(0).unsqueeze(0).float(),
                size=(h_orig, w_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

        # Normalize correlation map
        corr_map = (corr_map - corr_map.min()) / (corr_map.max() - corr_map.min() + 1e-8)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(original_img.astype(np.uint8))
        axes[0].set_title('Original Image', fontsize=20, fontweight='bold')
        axes[0].axis('off')

        # Draw exemplar boxes if provided
        if exemplar_boxes is not None:
            for box in exemplar_boxes:
                # Handle different box formats:
                # Format 1: [[x1, y1], [x2, y2]] - nested list format
                # Format 2: [x1, y1, x2, y2] - flat list format
                box_arr = np.asarray(box)
                x_arr = box_arr[:, 0]
                y_arr = box_arr[:, 1]
                x1, y1 = np.min(x_arr), np.min(y_arr)
                x2, y2 = np.max(x_arr), np.max(y_arr)

                # Only draw if box has valid dimensions
                if abs(x2 - x1) > 1 and abs(y2 - y1) > 1:
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                            linewidth=2, edgecolor='red', facecolor='none')
                    axes[0].add_patch(rect)
                else:
                    print(f"Warning: Box too small or invalid, skipping. Box: ({x1}, {y1}, {x2}, {y2})")

        # Correlation map as heatmap
        im1 = axes[1].imshow(corr_map, cmap=self.cmap, interpolation='bilinear')
        axes[1].set_title('Similarity Map (Heatmap)', fontsize=18, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Overlay correlation map on original image
        corr_colored = self.cmap(corr_map)[:, :, :3] * 255
        overlay = 0.6 * corr_colored + 0.4 * original_img
        axes[2].imshow(overlay.astype(np.uint8))
        axes[2].set_title('Overlay Visualization', fontsize=18, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def visualize_dynamic_weights(self, dynamic_weights, exemplar_indices=None,
                                   save_path=None, title="Dynamic Channel Attention Weights"):
        """
        Visualize dynamic channel attention weights (Fig. 3 in paper)

        Args:
            dynamic_weights: Dynamic weights tensor [B, num_exemplars, proj_dim] or [num_exemplars, proj_dim]
            exemplar_indices: Indices or labels for exemplars
            save_path: Path to save the visualization
            title: Title for the figure
        """
        if isinstance(dynamic_weights, torch.Tensor):
            dynamic_weights = dynamic_weights.cpu().numpy()

        if dynamic_weights.ndim == 3:
            dynamic_weights = dynamic_weights[0]  # Take first batch

        num_exemplars, proj_dim = dynamic_weights.shape

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, num_exemplars * 0.8)))

        im = ax.imshow(dynamic_weights, cmap='RdYlBu_r', aspect='auto',
                      interpolation='nearest', vmin=-1, vmax=1)

        # Set labels
        ax.set_xlabel('Channel Dimension', fontsize=15, fontweight='bold')
        ax.set_ylabel('Exemplar Index', fontsize=15, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')

        # Set ticks
        ax.set_yticks(range(num_exemplars))
        if exemplar_indices is not None:
            ax.set_yticklabels([f'Exemplar {i}' for i in exemplar_indices])
        else:
            ax.set_yticklabels([f'Exemplar {i}' for i in range(num_exemplars)])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def visualize_self_attention(self, attention_map, original_img, query_points=None,
                                save_path=None, title="Self-Attention Visualization"):
        """
        Visualize self-attention maps (Fig. 5 in paper)

        Args:
            attention_map: Attention map tensor [B, N, N] where N = H*W
            original_img: Original image array [H, W, 3]
            query_points: List of query point coordinates [(x, y), ...]
            save_path: Path to save the visualization
            title: Title for the figure
        """
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.cpu().numpy()

        if attention_map.ndim == 3:
            attention_map = attention_map[0]  # Take first batch

        h_orig, w_orig = original_img.shape[:2]
        N = attention_map.shape[0]
        print(f"Visualizing self-attention with N={N} (attention map shape: {attention_map.shape})")
        # Get the actual length of attention vector (may differ from N)
        # Check the shape of the first attention vector to determine actual dimensions
        if attention_map.shape[1] != N:
            # If attention_map is [N, M] where M != N, use M for reshaping
            attn_length = attention_map.shape[1]
        else:
            attn_length = N
        
        # Calculate attention map spatial dimensions based on actual length
        h_attn = int(np.sqrt(attn_length))
        w_attn = attn_length // h_attn
        
        # Adjust if not perfect match
        if h_attn * w_attn != attn_length:
            # Find closest dimensions
            h_attn = int(np.sqrt(attn_length))
            w_attn = (attn_length + h_attn - 1) // h_attn  # Ceiling division

        # Select query points to visualize
        if query_points is None:
            # Select a few representative query points
            num_queries = min(6, N)
            query_indices = np.linspace(0, N-1, num_queries, dtype=int)
        else:
            # Convert query points to indices
            query_indices = []
            for x, y in query_points:
                idx = int(y * h_attn / h_orig) * w_attn + int(x * w_attn / w_orig)
                query_indices.append(min(idx, attn_length-1))

        num_queries = len(query_indices)
        cols = 3
        rows = (num_queries + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, query_idx in enumerate(query_indices):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            # Get attention for this query point
            attn_vector = attention_map[query_idx].ravel()  # Flatten to 1D
            attn_size = attn_vector.size
            
            # Find the best factor pair for reshaping
            # Start from sqrt and find factors that divide evenly
            h_attn_actual = int(np.sqrt(attn_size))
            w_attn_actual = attn_size // h_attn_actual
            
            # If not perfect match, find the closest factor pair
            if h_attn_actual * w_attn_actual != attn_size:
                # Find factors of attn_size
                best_h, best_w = h_attn_actual, w_attn_actual
                min_diff = abs(h_attn_actual - w_attn_actual)
                
                # Try to find factor pairs
                for h in range(int(np.sqrt(attn_size)), 0, -1):
                    if attn_size % h == 0:
                        w = attn_size // h
                        diff = abs(h - w)
                        if diff < min_diff:
                            min_diff = diff
                            best_h, best_w = h, w
                            break
                
                # If no perfect factor found, use the closest dimensions
                if best_h * best_w != attn_size:
                    # Use dimensions that require minimal padding
                    h_attn_actual = int(np.sqrt(attn_size))
                    w_attn_actual = (attn_size + h_attn_actual - 1) // h_attn_actual  # Ceiling division
                    # Pad with zeros if needed
                    if h_attn_actual * w_attn_actual > attn_size:
                        attn_padded = np.zeros(h_attn_actual * w_attn_actual)
                        attn_padded[:attn_size] = attn_vector
                        attn = attn_padded.reshape(h_attn_actual, w_attn_actual)
                    else:
                        attn = attn_vector[:h_attn_actual * w_attn_actual].reshape(h_attn_actual, w_attn_actual)
                else:
                    h_attn_actual, w_attn_actual = best_h, best_w
                    attn = attn_vector.reshape(h_attn_actual, w_attn_actual)
            else:
                attn = attn_vector.reshape(h_attn_actual, w_attn_actual)

            # Resize to original image size
            attn_tensor = torch.from_numpy(attn).unsqueeze(0).unsqueeze(0).float()
            attn_resized = F.interpolate(
                attn_tensor,
                size=(h_orig, w_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

            # Normalize
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

            # Overlay on original image
            attn_colored = self.cmap(attn_resized)[:, :, :3] * 255
            overlay = 0.5 * attn_colored + 0.5 * original_img

            ax.imshow(overlay.astype(np.uint8))
            ax.set_title(f'Query Point {query_idx}', fontsize=12)
            ax.axis('off')

        # Hide unused subplots
        for idx in range(num_queries, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def visualize_density_map(self, pred_density, gt_density, original_img,
                             pred_count, gt_count, save_path=None,exemplar_boxes=None, title="Density Map Comparison"):
        """
        Visualize predicted and ground truth density maps (Fig. 6 in paper)

        Args:
            pred_density: Predicted density map [B, 1, H, W] or [H, W]
            gt_density: Ground truth density map [B, 1, H, W] or [H, W]
            original_img: Original image array [H, W, 3]
            pred_count: Predicted count
            gt_count: Ground truth count
            save_path: Path to save the visualization
            exemplar_boxes: List of exemplar boxes [[x1, y1, x2, y2], ...]
            title: Title for the figure
        """
        # Convert to numpy
        if isinstance(pred_density, torch.Tensor):
            if pred_density.dim() == 4:
                pred_density = pred_density[0, 0]
            elif pred_density.dim() == 3:
                pred_density = pred_density[0]
            pred_density = pred_density.cpu().numpy()

        if isinstance(gt_density, torch.Tensor):
            if gt_density.dim() == 4:
                gt_density = gt_density[0, 0]
            elif gt_density.dim() == 3:
                gt_density = gt_density[0]
            gt_density = gt_density.cpu().numpy()

        h_orig, w_orig = original_img.shape[:2]
        h_pred, w_pred = pred_density.shape
        h_gt, w_gt = gt_density.shape

        # Resize to original image size
        if h_pred != h_orig or w_pred != w_orig:
            pred_density = F.interpolate(
                torch.from_numpy(pred_density).unsqueeze(0).unsqueeze(0).float(),
                size=(h_orig, w_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

        if h_gt != h_orig or w_gt != w_orig:
            gt_density = F.interpolate(
                torch.from_numpy(gt_density).unsqueeze(0).unsqueeze(0).float(),
                size=(h_orig, w_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()

        # Normalize
        pred_density_norm = (pred_density - pred_density.min()) / (pred_density.max() - pred_density.min() + 1e-8)
        gt_density_norm = (gt_density - gt_density.min()) / (gt_density.max() - gt_density.min() + 1e-8)

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: Original, GT Density, GT Overlay
        axes[0, 0].imshow(original_img.astype(np.uint8))
        axes[0, 0].set_title('Original Image', fontsize=18, fontweight='bold')
        axes[0, 0].axis('off')

        im1 = axes[0, 1].imshow(gt_density_norm, cmap=self.cmap, interpolation='bilinear')
        axes[0, 1].set_title(f'Ground Truth Density Map (Count: {gt_count:.1f})',
                            fontsize=18, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        gt_colored = self.cmap(gt_density_norm)[:, :, :3] * 255
        overlay_gt = 0.6 * gt_colored + 0.4 * original_img
        axes[0, 2].imshow(overlay_gt.astype(np.uint8))
        axes[0, 2].set_title('GT Density Overlay', fontsize=18, fontweight='bold')
        axes[0, 2].axis('off')

        # Row 2: Original, Pred Density, Pred Overlay
        axes[1, 0].imshow(original_img.astype(np.uint8))
        axes[1, 0].set_title('Original Image', fontsize=18, fontweight='bold')
        axes[1, 0].axis('off')

        im2 = axes[1, 1].imshow(pred_density_norm, cmap=self.cmap, interpolation='bilinear')
        axes[1, 1].set_title(f'Predicted Density Map (Count: {pred_count:.1f})',
                            fontsize=18, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

        pred_colored = self.cmap(pred_density_norm)[:, :, :3] * 255
        overlay_pred = 0.6 * pred_colored + 0.4 * original_img
        axes[1, 2].imshow(overlay_pred.astype(np.uint8))
        axes[1, 2].set_title('Predicted Density Overlay', fontsize=18, fontweight='bold')
        axes[1, 2].axis('off')
        # Draw exemplar boxes if provided
        if exemplar_boxes is not None:
            for box in exemplar_boxes:
                # Handle different box formats:
                # Format 1: [[x1, y1], [x2, y2]] - nested list format
                box_arr = np.asarray(box)
                x_arr = box_arr[:, 0]
                y_arr = box_arr[:, 1]
                x1, y1 = np.min(x_arr), np.min(y_arr)
                x2, y2 = np.max(x_arr), np.max(y_arr)

                # Only draw if box has valid dimensions
                if abs(x2 - x1) > 1 and abs(y2 - y1) > 1:
                    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              linewidth=2, edgecolor='red', facecolor='none')
                    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                              linewidth=2, edgecolor='red', facecolor='none')
                    axes[0, 0].add_patch(rect1)
                    axes[1, 0].add_patch(rect2)
                else:
                    print(f"Warning: Box too small or invalid, skipping. Box: ({x1}, {y1}, {x2}, {y2})")

        # Add error information
        error = abs(pred_count - gt_count)
        error_pct = (error / (gt_count + 1e-8)) * 100
        plt.suptitle(f'{title}\nMAE: {error:.2f}',
                    fontsize=20, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def visualize_similarity_loss(self, corr_map, pt_map, save_path=None,
                                  title="Similarity Loss Visualization"):
        """
        Visualize similarity loss computation (Fig. 4 in paper)
        Shows positive (target) and negative (background) regions

        Args:
            corr_map: Correlation map [B, H, W] or [H, W]
            pt_map: Point map indicating target locations [B, 1, H, W] or [H, W]
            save_path: Path to save the visualization
            title: Title for the figure
        """
        if isinstance(corr_map, torch.Tensor):
            if corr_map.dim() == 4:
                corr_map = corr_map[0, 0]
            elif corr_map.dim() == 3:
                corr_map = corr_map[0]
            corr_map = corr_map.cpu().numpy()

        if isinstance(pt_map, torch.Tensor):
            if pt_map.dim() == 4:
                pt_map = pt_map[0, 0]
            elif pt_map.dim() == 3:
                pt_map = pt_map[0]
            pt_map = pt_map.cpu().numpy()

        # Create positive and negative masks
        # Positive: regions with targets (pt_map > 0)
        # Negative: regions without targets (pt_map == 0)
        pos_mask = pt_map > 0
        neg_mask = pt_map == 0

        # Resize masks to match corr_map if needed
        if pos_mask.shape != corr_map.shape:
            pos_mask = F.interpolate(
                torch.from_numpy(pos_mask).unsqueeze(0).unsqueeze(0).float(),
                size=corr_map.shape,
                mode='nearest'
            ).squeeze().numpy() > 0.5

        if neg_mask.shape != corr_map.shape:
            neg_mask = F.interpolate(
                torch.from_numpy(neg_mask).unsqueeze(0).unsqueeze(0).float(),
                size=corr_map.shape,
                mode='nearest'
            ).squeeze().numpy() > 0.5

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Correlation map
        im1 = axes[0].imshow(corr_map, cmap=self.cmap, interpolation='bilinear')
        axes[0].set_title('Similarity Map', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Positive regions
        pos_vis = corr_map.copy()
        pos_vis[~pos_mask] = 0
        im2 = axes[1].imshow(pos_vis, cmap='Greens', interpolation='bilinear')
        axes[1].set_title(f'Positive Regions (Signal)\n{pos_mask.sum()} pixels',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Negative regions
        neg_vis = corr_map.copy()
        neg_vis[~neg_mask] = 0
        im3 = axes[2].imshow(neg_vis, cmap='Reds', interpolation='bilinear')
        axes[2].set_title(f'Negative Regions (Noise)\n{neg_mask.sum()} pixels',
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def visualize_feature_maps(self, features, original_img, num_channels=8,
                               save_path=None, title="Feature Map Visualization"):
        """
        Visualize intermediate feature maps

        Args:
            features: Feature tensor [B, C, H, W] or [C, H, W]
            original_img: Original image array [H, W, 3]
            num_channels: Number of channels to visualize
            save_path: Path to save the visualization
            title: Title for the figure
        """
        if isinstance(features, torch.Tensor):
            if features.dim() == 4:
                features = features[0]
            features = features.cpu().numpy()

        C, H, W = features.shape
        num_channels = min(num_channels, C)

        # Select channels with highest variance
        channel_vars = np.var(features.reshape(C, -1), axis=1)
        top_channels = np.argsort(channel_vars)[-num_channels:][::-1]

        cols = 4
        rows = (num_channels + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for idx, ch_idx in enumerate(top_channels):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            feat_map = features[ch_idx]

            # Resize to original image size
            h_orig, w_orig = original_img.shape[:2]
            if H != h_orig or W != w_orig:
                feat_tensor = torch.from_numpy(feat_map).unsqueeze(0).unsqueeze(0).float()
                feat_map = F.interpolate(
                    feat_tensor,
                    size=(h_orig, w_orig),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()

            # Normalize
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)

            im = ax.imshow(feat_map, cmap='viridis', interpolation='bilinear')
            ax.set_title(f'Channel {ch_idx}', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Hide unused subplots
        for idx in range(num_channels, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def visualize_similarity_distribution(self, corr_map, save_path=None,
                                          title="Similarity Distribution"):
        """
        Visualize the distribution of similarity values

        Args:
            corr_map: Correlation map [B, H, W] or [H, W]
            save_path: Path to save the visualization
            title: Title for the figure
        """
        if isinstance(corr_map, torch.Tensor):
            if corr_map.dim() == 4:
                corr_map = corr_map[0, 0]
            elif corr_map.dim() == 3:
                corr_map = corr_map[0]
            corr_map = corr_map.cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(corr_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Similarity Value', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Similarity Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Statistics
        mean_val = corr_map.mean()
        std_val = corr_map.std()
        min_val = corr_map.min()
        max_val = corr_map.max()
        median_val = np.median(corr_map)

        stats_text = f'Mean: {mean_val:.4f}\n'
        stats_text += f'Std: {std_val:.4f}\n'
        stats_text += f'Min: {min_val:.4f}\n'
        stats_text += f'Max: {max_val:.4f}\n'
        stats_text += f'Median: {median_val:.4f}'

        axes[1].text(0.1, 0.5, stats_text, fontsize=12,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        axes[1].set_title('Statistics', fontsize=14, fontweight='bold')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            return fig

    def create_comprehensive_visualization(self, outputs, original_img, exemplar_boxes=None,
                                          pt_map=None, gt_density=None, gt_count=None, pred_count=None,
                                          file_name="comprehensive_vis", save_dir=None):
        """
        Create a comprehensive visualization combining all visualizations

        Args:
            outputs: Dictionary containing model outputs
            original_img: Original image array
            exemplar_boxes: Exemplar boxes
            pt_map: Point map for ground truth (used for similarity loss visualization)
            gt_density: Ground truth density map (used for density map visualization)
            gt_count: Ground truth count
            pred_count: Predicted count
            file_name: Base name for saved files
            save_dir: Directory to save visualizations
        """
        if save_dir is None:
            save_dir = self.output_dir

        os.makedirs(save_dir, exist_ok=True)

        # 1. Correlation Map Visualization
        if 'corr_map' in outputs:
            corr_map = outputs['corr_map']
            if isinstance(corr_map, torch.Tensor):
                if corr_map.dim() == 4:  # [B, H, W, num_exemplars] or [B, num_exemplars, H, W]
                    if corr_map.shape[1] < corr_map.shape[2]:  # [B, H, W, num_exemplars]
                        corr_map = corr_map[0].mean(dim=-1)  # Average over exemplars
                    else:  # [B, num_exemplars, H, W]
                        corr_map = corr_map[0].mean(dim=0)  # Average over exemplars
                elif corr_map.dim() == 3:
                    if corr_map.shape[0] == 1:  # [1, H, W]
                        corr_map = corr_map[0]
                    else:  # [num_exemplars, H, W] or [H, W, num_exemplars]
                        corr_map = corr_map.mean(dim=0) if corr_map.shape[0] < corr_map.shape[1] else corr_map.mean(dim=-1)
            self.visualize_correlation_map(
                corr_map, original_img, exemplar_boxes,
                save_path=os.path.join(save_dir, f'{file_name}_correlation_map.png')
            )

        # 2. Dynamic Weights Visualization
        if 'dynamic_weights' in outputs and outputs['dynamic_weights'] is not None:
            self.visualize_dynamic_weights(
                outputs['dynamic_weights'],
                save_path=os.path.join(save_dir, f'{file_name}_dynamic_weights.png')
            )

        # 3. Self-Attention Visualization
        if 'attention_maps' in outputs and outputs['attention_maps'] is not None:
            attention_maps = outputs['attention_maps']
            if len(attention_maps) > 0:
                self.visualize_self_attention(
                    attention_maps[0], original_img,
                    save_path=os.path.join(save_dir, f'{file_name}_self_attention.png')
                )

        # 4. Density Map Visualization
        if 'density_map' in outputs and gt_count is not None:
            pred_density = outputs['density_map']
            # Use provided GT density map, or fallback to pt_map, or create dummy
            if gt_density is not None:
                # Use the provided GT density map
                pass
            elif pt_map is not None:
                # Fallback to pt_map if gt_density not provided
                gt_density = pt_map
            else:
                # Create dummy GT density if neither provided
                pred_shape = pred_density[0, 0].cpu().numpy() if isinstance(pred_density, torch.Tensor) else pred_density
                gt_density = np.zeros_like(pred_shape)

            self.visualize_density_map(
                pred_density, gt_density, original_img,
                pred_count or pred_density.sum().item() if isinstance(pred_density, torch.Tensor) else pred_density.sum(),
                gt_count,
                exemplar_boxes=exemplar_boxes,
                save_path=os.path.join(save_dir, f'{file_name}_density_map.png')
            )

        # 5. Similarity Loss Visualization
        if 'corr_map' in outputs and pt_map is not None:
            self.visualize_similarity_loss(
                outputs['corr_map'], pt_map,
                save_path=os.path.join(save_dir, f'{file_name}_similarity_loss.png')
            )

        # 6. Feature Maps Visualization
        if 'refined_features' in outputs:
            self.visualize_feature_maps(
                outputs['refined_features'], original_img,
                save_path=os.path.join(save_dir, f'{file_name}_feature_maps.png')
            )

        # 7. Similarity Distribution
        if 'corr_map' in outputs:
            corr_map = outputs['corr_map']
            if isinstance(corr_map, torch.Tensor):
                if corr_map.dim() == 4:  # [B, H, W, num_exemplars] or [B, num_exemplars, H, W]
                    if corr_map.shape[1] < corr_map.shape[2]:  # [B, H, W, num_exemplars]
                        corr_map = corr_map[0].mean(dim=-1)  # Average over exemplars
                    else:  # [B, num_exemplars, H, W]
                        corr_map = corr_map[0].mean(dim=0)  # Average over exemplars
                elif corr_map.dim() == 3:
                    if corr_map.shape[0] == 1:  # [1, H, W]
                        corr_map = corr_map[0]
                    else:  # [num_exemplars, H, W] or [H, W, num_exemplars]
                        corr_map = corr_map.mean(dim=0) if corr_map.shape[0] < corr_map.shape[1] else corr_map.mean(dim=-1)
            self.visualize_similarity_distribution(
                corr_map,
                save_path=os.path.join(save_dir, f'{file_name}_similarity_distribution.png')
            )
