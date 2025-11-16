"""
直接运行的可视化脚本 - 一键生成所有可视化结果
只需修改下面的配置路径即可使用
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from config import cfg
from FSC147_dataset import build_dataset, batch_collate_fn
from models import build_model
from util.visualization import Visualizer

# ==================== 配置区域 - 请根据你的实际情况修改 ====================
# 配置文件路径
CONFIG_FILE = "config/bmnet+_fsc147.yaml"

# 模型检查点路径
CHECKPOINT_PATH = "checkpoints/best/model_best.pth"

# 输出目录
OUTPUT_DIR = "./visualizations"

# 可视化样本数量（-1表示所有样本，或指定具体数字如3）
NUM_SAMPLES = 1

# 颜色映射（可选：'jet', 'viridis', 'hot', 'coolwarm'等）
COLORMAP = 'jet'
# ===========================================================================


def load_model(checkpoint_path, device):
    """加载模型"""
    print(f"正在加载模型: {checkpoint_path}")
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的检查点格式
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✓ 模型加载成功!")
    return model


def get_exemplar_boxes(annotation_file, file_name, scale_factor=1.0):
    """从标注文件获取示例框"""
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        if file_name not in annotations:
            return None
        
        boxes = np.array(annotations[file_name]['box_examples_coordinates'])
        boxes = boxes * scale_factor
        return boxes[:3]  # 返回前3个示例
    except:
        return None


def main():
    print("=" * 80)
    print("BMNet/BMNet+ 可视化工具 - 一键生成所有可视化结果")
    print("=" * 80)
    
    # 加载配置文件
    print(f"\n加载配置文件: {CONFIG_FILE}")
    cfg.merge_from_file(CONFIG_FILE)
    cfg.freeze()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ 错误: 找不到模型文件: {CHECKPOINT_PATH}")
        print("请修改脚本中的 CHECKPOINT_PATH 变量")
        return
    model = load_model(CHECKPOINT_PATH, device)
    
    # 构建数据集
    print(f"\n构建数据集: {cfg.DIR.dataset}")
    if not os.path.exists(cfg.DIR.dataset):
        print(f"❌ 错误: 找不到数据集目录: {cfg.DIR.dataset}")
        print("请检查配置文件中的数据集路径")
        return
    
    dataset = build_dataset(cfg, is_train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.TRAIN.num_workers,
        collate_fn=batch_collate_fn
    )
    print(f"✓ 数据集加载成功: {len(dataset)} 个样本")
    
    # 创建可视化器
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visualizer = Visualizer(OUTPUT_DIR, cmap=COLORMAP)
    print(f"\n可视化结果将保存到: {OUTPUT_DIR}")
    
    # 加载标注文件（用于示例框）
    annotation_file = os.path.join(cfg.DIR.dataset, 'annotation_FSC147_384.json')
    has_annotations = os.path.exists(annotation_file)
    if has_annotations:
        print(f"✓ 找到标注文件: {annotation_file}")
    else:
        print(f"⚠ 未找到标注文件，将跳过示例框可视化")
    
    # 处理样本
    num_samples = NUM_SAMPLES if NUM_SAMPLES > 0 else len(data_loader)
    print(f"\n开始处理 {num_samples} 个样本...")
    print("-" * 80)
    
    mae_sum = 0
    mse_sum = 0
    processed_count = 0
    
    for idx, sample in enumerate(data_loader):
        if idx >= num_samples:
            break
        
        try:
            img, patches, targets = sample
            img = img.to(device)
            patches['patches'] = patches['patches'].to(device)
            patches['scale_embedding'] = patches['scale_embedding'].to(device)
            
            # 获取文件名
            file_name = dataset.data_list[idx][0]
            print(f"\n[{idx+1}/{num_samples}] 处理: {file_name}")
            
            # 加载原始图像
            image_path = os.path.join(cfg.DIR.dataset, 'images_384_VarV2', file_name)
            if not os.path.exists(image_path):
                # 尝试其他可能的路径
                alt_paths = [
                    os.path.join(cfg.DIR.dataset, 'images', file_name),
                    os.path.join(cfg.DIR.dataset, file_name),
                ]
                found = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        found = True
                        break
                
                if not found:
                    print(f"  ⚠ 警告: 找不到图像文件，跳过...")
                    continue
            
            original_img = Image.open(image_path).convert("RGB")
            original_img = np.array(original_img)
            
            # 计算缩放因子
            h_orig, w_orig = original_img.shape[:2]
            h_model, w_model = img.shape[-2:]
            scale_factor = min(h_orig / h_model, w_orig / w_model)
            
            # 获取示例框
            exemplar_boxes = None
            if has_annotations:
                exemplar_boxes = get_exemplar_boxes(annotation_file, file_name, scale_factor)
            
            # 运行模型（获取中间输出）
            with torch.no_grad():
                outputs = model(img, patches, is_train=False, return_intermediate=True)
            
            # 获取计数结果
            pred_count = outputs['density_map'].sum().item()
            gt_count = targets['gtcount'].item()
            
            # 计算指标
            error = abs(pred_count - gt_count)
            mae_sum += error
            mse_sum += error ** 2
            processed_count += 1
            
            print(f"  真实计数: {gt_count:.1f}, 预测计数: {pred_count:.1f}, 误差: {error:.2f}")
            
            # 处理相关性图 - 转换为 [B, H, W] 格式
            corr_map = outputs['corr_map']
            if isinstance(corr_map, torch.Tensor):
                if corr_map.dim() == 3:  # [B, H*W, num_exemplars] 或 [B, H, W]
                    bs, first_dim, second_dim = corr_map.shape
                    if first_dim > second_dim:  # [B, H*W, num_exemplars]
                        # 需要重塑为 [B, H, W, num_exemplars] 然后平均
                        h, w = outputs['refined_features'].shape[-2:]
                        corr_map = corr_map.view(bs, h, w, second_dim).mean(dim=-1)  # [B, H, W]
                    # 否则已经是 [B, H, W]
            
            # 准备可视化输出
            vis_outputs = {
                'density_map': outputs['density_map'],
                'corr_map': corr_map,
                'attention_maps': outputs.get('attention_maps'),
                'dynamic_weights': outputs.get('dynamic_weights'),
                'refined_features': outputs.get('refined_features'),
                'patch_features': outputs.get('patch_features'),
                'query_features': outputs.get('query_features')
            }
            
            # 获取点图和真实密度图
            pt_map = targets['pt_map'].cpu().numpy()[0, 0] if 'pt_map' in targets else None
            gt_density = targets['density_map'].cpu().numpy()[0, 0] if 'density_map' in targets else None
            
            # 创建综合可视化
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
                save_dir=OUTPUT_DIR
            )
            
            print(f"  ✓ 已保存可视化结果: {base_name}")
            
        except Exception as e:
            print(f"  ❌ 处理样本时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印总结
    if processed_count > 0:
        mae = mae_sum / processed_count
        mse = (mse_sum / processed_count) ** 0.5
        print("\n" + "=" * 80)
        print("处理完成!")
        print(f"  处理样本数: {processed_count}")
        print(f"  平均绝对误差 (MAE): {mae:.2f}")
        print(f"  均方根误差 (RMSE): {mse:.2f}")
        print(f"  可视化结果保存位置: {OUTPUT_DIR}")
        print("=" * 80)
    else:
        print("\n❌ 没有成功处理任何样本，请检查配置路径")


if __name__ == '__main__':
    main()

