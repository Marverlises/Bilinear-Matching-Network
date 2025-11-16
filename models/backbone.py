# ------------------------------------------------------------------------
# Modified from UP-DETR  https://github.com/dddzg/up-detr
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not found. ViT backbone will not be available.")

#from util.misc import NestedTensor
#from .position_encoding import build_position_encoding
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layer: str):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            #if not train_backbone:
                parameter.requires_grad_(False)
        
        return_layers = {return_layer: '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        """supports both NestedTensor and torch.Tensor
        """
        out = self.body(tensor_list)
        return out['0']

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_layer: str,
                 frozen_bn: bool,
                 dilation: bool):
        
        if frozen_bn:
            backbone = getattr(torchvision.models, name)(
                               replace_stride_with_dilation=[False, False, dilation],
                               pretrained=True, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(torchvision.models, name)(
                               replace_stride_with_dilation=[False, False, dilation],
                               pretrained=True)
            
        # load the SwAV pre-training model from the url instead of supervised pre-training model
        if name == 'resnet50':
            checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            backbone.load_state_dict(state_dict, strict=False)
            #pass
        if name in ('resnet18', 'resnet34'):
            num_channels = 512
        else:
            if return_layer == 'layer3':
                num_channels = 1024
            else:
                num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_layer)


class ViTBackbone(nn.Module):
    """Vision Transformer (ViT) backbone."""
    
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224',
                 train_backbone: bool = True,
                 pretrained: bool = True):
        """
        Args:
            model_name: ViT model name from timm (e.g., 'vit_base_patch16_224', 'vit_large_patch16_224')
            train_backbone: Whether to train the backbone
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for ViT backbone. Install it with: pip install timm")
        
        # Load ViT model from timm
        # Try to create model with dynamic image size support
        try:
            # Some timm versions support dynamic_img_size parameter
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool='',  # Don't pool, we need all tokens
                dynamic_img_size=True,  # Allow dynamic input sizes
            )
        except TypeError:
            # If dynamic_img_size is not supported, create normally and modify patch_embed
            # Try to create with a larger img_size that can accommodate most inputs
            # Or create normally and patch the forward method
            try:
                # Try creating with img_size parameter set to a large value
                self.vit = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool='',
                    img_size=384,  # Set to match common input size
                )
            except (TypeError, ValueError):
                # If img_size parameter doesn't work, create normally and patch
                self.vit = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool='',
                )
                # Patch patch_embed to support dynamic input sizes
                # Store original forward method and img_size
                original_forward = self.vit.patch_embed.forward
                original_img_size = getattr(self.vit.patch_embed, 'img_size', None)
                
                def patched_forward(x):
                    """Patched forward that bypasses img_size check"""
                    B, C, H, W = x.shape
                    # Temporarily modify img_size to match input to bypass assertion
                    # timm checks: _assert(H == self.img_size[0], ...)
                    if hasattr(self.vit.patch_embed, 'img_size') and self.vit.patch_embed.img_size is not None:
                        # Store original value
                        saved_img_size = self.vit.patch_embed.img_size
                        # Set to match input dimensions (as tuple/list)
                        if isinstance(saved_img_size, (list, tuple)):
                            self.vit.patch_embed.img_size = [H, W]
                        else:
                            self.vit.patch_embed.img_size = (H, W)
                    try:
                        result = original_forward(x)
                    finally:
                        # Restore original img_size
                        if hasattr(self.vit.patch_embed, 'img_size'):
                            self.vit.patch_embed.img_size = original_img_size
                    return result
                
                self.vit.patch_embed.forward = patched_forward
        
        # Get model configuration
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]
        # Store original img_size if available, otherwise use None
        if hasattr(self.vit.patch_embed, 'img_size') and self.vit.patch_embed.img_size is not None:
            self.img_size = self.vit.patch_embed.img_size[0] if isinstance(self.vit.patch_embed.img_size, (list, tuple)) else self.vit.patch_embed.img_size
        else:
            self.img_size = None
        
        # Freeze parameters if not training backbone
        if not train_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        self.num_channels = self.embed_dim
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 3, H, W]
        Returns:
            Feature map of shape [batch_size, embed_dim, H', W']
        """
        B, C, H, W = x.shape
        
        # Forward through ViT
        # Output shape: [batch_size, num_patches + 1, embed_dim] (includes CLS token)
        x = self.vit.forward_features(x)
        
        # Remove CLS token (first token) if present
        # Shape: [batch_size, num_patches, embed_dim]
        if x.shape[1] > 1:  # Check if CLS token exists
            x = x[:, 1:]
        
        # Calculate spatial dimensions
        # For ViT, the number of patches depends on input size and patch size
        num_patches = x.shape[1]
        
        # Calculate expected spatial dimensions
        # ViT typically expects square inputs, but we handle rectangular inputs
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        expected_patches = h_patches * w_patches
        
        # If the number of patches matches expected, use calculated dimensions
        if num_patches == expected_patches:
            # Use calculated dimensions
            pass
        else:
            # Infer dimensions from actual number of patches
            # Try to find factors that make sense
            h_patches = int(num_patches ** 0.5)
            while num_patches % h_patches != 0:
                h_patches -= 1
            w_patches = num_patches // h_patches
        
        # Reshape to feature map format: [batch_size, embed_dim, h_patches, w_patches]
        x = x.permute(0, 2, 1).contiguous()  # [B, embed_dim, num_patches]
        x = x.view(B, self.embed_dim, h_patches, w_patches)
        
        return x


def build_backbone(cfg):
    train_backbone = cfg.TRAIN.lr_backbone > 0
    backbone_name = cfg.MODEL.backbone.lower()
    
    # Check if it's a ViT model
    if backbone_name.startswith('vit') or 'transformer' in backbone_name:
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for ViT backbone. Install it with: pip install timm")
        
        # Default ViT model names
        vit_model_map = {
            'vit': 'vit_base_patch16_224',
            'vit_base': 'vit_base_patch16_224',
            'vit_large': 'vit_large_patch16_224',
            'vit_small': 'vit_small_patch16_224',
            'vit_tiny': 'vit_tiny_patch16_224',
        }
        
        # Use provided name or default
        model_name = vit_model_map.get(backbone_name, backbone_name)
        
        # Check if it's a valid timm model name
        try:
            available_models = timm.list_models('*vit*')
            if model_name not in available_models and model_name not in vit_model_map.values():
                print(f"Warning: {model_name} may not be a valid ViT model. Using vit_base_patch16_224 as default.")
                model_name = 'vit_base_patch16_224'
        except:
            # If list_models fails, try to use the model name directly
            # timm will raise an error if the model doesn't exist
            pass
        
        backbone = ViTBackbone(
            model_name=model_name,
            train_backbone=train_backbone,
            pretrained=cfg.MODEL.pretrain if hasattr(cfg.MODEL, 'pretrain') else True
        )
    else:
        # ResNet backbone (original)
        backbone = Backbone(
            cfg.MODEL.backbone, 
            train_backbone, 
            cfg.MODEL.backbone_layer, 
            cfg.MODEL.fix_bn, 
            cfg.MODEL.dilation
        )
    
    return backbone

if __name__ == '__main__':
    backbone = Backbone('resnet50',
                        train_backbone=True,
                        return_layer='layer3',
                        frozen_bn=False,
                        dilation=False)
    
    inputs = torch.rand(5,3,256,256)
    outputs = backbone(inputs)
