"""
Class Agnostic Counting Web Visualization Application
Provides image upload and visualization functionality
"""
import os
import io
import base64
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import torch
from werkzeug.utils import secure_filename

import sys
import os

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import cfg
from models import build_model
from util.visualization import Visualizer

# Import image processing functions (need to import from web_app package)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_processor import process_image_for_inference

# Get absolute path of web_app directory
web_app_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, 
            template_folder=os.path.join(web_app_dir, 'templates'),
            static_folder=os.path.join(web_app_dir, 'static'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'results')
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables to store model
model = None
device = None
visualizer = None


def load_model_once():
    """Load model (only load once)"""
    global model, device, visualizer
    
    if model is not None:
        return model, device, visualizer
    
    # Load configuration
    config_file = os.environ.get('CONFIG_FILE', 'config/bmnet+_fsc147.yaml')
    checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/best/model_best.pth')
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
    
    print(f"Loading config file: {config_file}")
    cfg.merge_from_file(config_file)
    cfg.freeze()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {checkpoint_path}")
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
    print("✓ Model loaded successfully!")
    
    # Create visualizer
    visualizer = Visualizer(app.config['OUTPUT_FOLDER'], cmap='jet')
    
    return model, device, visualizer


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/preview', methods=['POST'])
def preview_boxes():
    """Preview exemplar box regions (crop and return images within boxes)"""
    try:
        # Check if file exists
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Filename is empty'}), 400
        
        # Get exemplar boxes (required)
        exemplar_boxes = None
        if 'exemplar_boxes' in request.form:
            try:
                exemplar_boxes = json.loads(request.form['exemplar_boxes'])
                if not exemplar_boxes or len(exemplar_boxes) == 0:
                    return jsonify({'error': 'Please provide at least 1 exemplar box'}), 400
            except Exception as e:
                return jsonify({'error': f'Exemplar box format error: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Missing exemplar boxes'}), 400
        
        # Read file content
        image_data = file.read()
        # If file object supports seek, reset pointer (doesn't affect subsequent requests as each request has independent file object)
        try:
            file.seek(0)
        except:
            pass
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        img_width, img_height = image.size
        
        # Crop each exemplar box region
        preview_images = []
        for i, box in enumerate(exemplar_boxes):
            x1, y1, x2, y2 = box
            # Ensure coordinates are within image bounds
            x1 = max(0, min(int(x1), img_width))
            y1 = max(0, min(int(y1), img_height))
            x2 = max(0, min(int(x2), img_width))
            y2 = max(0, min(int(y2), img_height))
            
            # Ensure valid region
            if x2 > x1 and y2 > y1:
                # Crop region
                cropped = image.crop((x1, y1, x2, y2))
                
                # Convert to base64
                buffer = io.BytesIO()
                cropped.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                preview_images.append({
                    'index': i + 1,
                    'image': f'data:image/png;base64,{img_str}',
                    'box': [x1, y1, x2, y2]
                })
        
        return jsonify({
            'success': True,
            'previews': preview_images
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': f'Error during preview: {error_msg}'}), 500


@app.route('/api/process', methods=['POST'])
def process_image():
    """Process uploaded image and generate visualizations"""
    try:
        # Check if file exists
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Filename is empty'}), 400
        
        # Get exemplar boxes (required)
        exemplar_boxes = None
        if 'exemplar_boxes' in request.form:
            try:
                exemplar_boxes = json.loads(request.form['exemplar_boxes'])
                if not exemplar_boxes or len(exemplar_boxes) == 0:
                    return jsonify({'error': 'Please provide at least 1 exemplar box. Exemplar boxes tell the model what objects to count.'}), 400
            except Exception as e:
                return jsonify({'error': f'Exemplar box format error: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Missing exemplar boxes. Please draw at least 1 exemplar box on the image to select target objects to count.'}), 400
        
        # Load model
        model, device, visualizer = load_model_once()
        
        # Save uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image (note: exemplar_boxes are in original image coordinates, no need to multiply by scale_factor)
        img_tensor, patches_dict, original_img, scale_factor = process_image_for_inference(
            filepath,
            exemplar_boxes=exemplar_boxes
        )
        
        # Move to device
        img_tensor = img_tensor.to(device)
        patches_dict['patches'] = patches_dict['patches'].to(device)
        patches_dict['scale_embedding'] = patches_dict['scale_embedding'].to(device)
        
        # Run model inference
        with torch.no_grad():
            outputs = model(img_tensor, patches_dict, is_train=False, return_intermediate=True)
        
        # Get predicted count
        pred_count = outputs['density_map'].sum().item()
        
        # Process correlation map
        corr_map = outputs['corr_map']
        if isinstance(corr_map, torch.Tensor):
            if corr_map.dim() == 3:  # [B, H*W, num_exemplars] or [B, H, W]
                bs, first_dim, second_dim = corr_map.shape
                if first_dim > second_dim:  # [B, H*W, num_exemplars]
                    h, w = outputs['refined_features'].shape[-2:]
                    corr_map = corr_map.view(bs, h, w, second_dim).mean(dim=-1)  # [B, H, W]
                corr_map = corr_map[0].cpu().numpy() if corr_map.shape[0] == 1 else corr_map.cpu().numpy()
            elif corr_map.dim() == 4:
                corr_map = corr_map[0].mean(dim=0).cpu().numpy() if corr_map.shape[1] < corr_map.shape[2] else corr_map[0].mean(dim=-1).cpu().numpy()
        
        # Prepare visualization outputs
        vis_outputs = {
            'density_map': outputs['density_map'],
            'corr_map': corr_map,
            'dynamic_weights': outputs.get('dynamic_weights'),
        }
        
        # Exemplar box coordinates (exemplar_boxes are already in original image coordinates, original_img is also original size, so use directly)
        display_exemplar_boxes = None
        if exemplar_boxes:
            display_exemplar_boxes = exemplar_boxes
        
        # Generate visualizations (only generate the three needed)
        base_name = os.path.splitext(filename)[0]
        timestamp = str(int(torch.randint(0, 1000000, (1,)).item()))
        file_prefix = f"{base_name}_{timestamp}"
        
        # 1. Correlation map
        visualizer.visualize_correlation_map(
            corr_map, original_img, display_exemplar_boxes,
            save_path=os.path.join(app.config['OUTPUT_FOLDER'], f'{file_prefix}_correlation_map.png')
        )
        
        # 2. Dynamic weights
        if vis_outputs['dynamic_weights'] is not None:
            visualizer.visualize_dynamic_weights(
                vis_outputs['dynamic_weights'],
                save_path=os.path.join(app.config['OUTPUT_FOLDER'], f'{file_prefix}_dynamic_weights.png')
            )
        
        # 3. Density map (no GT, only show prediction)
        pred_density = outputs['density_map'][0, 0].cpu().numpy()
        gt_density = np.zeros_like(pred_density)  # Create empty GT density map
        visualizer.visualize_density_map(
            outputs['density_map'],
            torch.from_numpy(gt_density).unsqueeze(0).unsqueeze(0),
            original_img,
            pred_count,
            0.0,  # No GT count, set to 0
            save_path=os.path.join(app.config['OUTPUT_FOLDER'], f'{file_prefix}_density_map.png'),
            exemplar_boxes=display_exemplar_boxes
        )
        
        # Return results
        result = {
            'success': True,
            'pred_count': round(pred_count, 2),
            'images': {
                'correlation_map': f'/static/results/{file_prefix}_correlation_map.png',
                'dynamic_weights': f'/static/results/{file_prefix}_dynamic_weights.png',
                'density_map': f'/static/results/{file_prefix}_density_map.png',
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': f'Error processing image: {error_msg}'}), 500


@app.route('/static/results/<filename>')
def serve_result(filename):
    """Serve generated visualization results"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


# Add static file route (for CSS and JS)
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    return send_from_directory(static_dir, filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Class Agnostic Counting Web Visualization Application')
    parser.add_argument('--config', type=str, default='config/bmnet+_fsc147.yaml',
                        help='Config file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best/model_best.pth',
                        help='Model checkpoint path')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['CONFIG_FILE'] = args.config
    os.environ['CHECKPOINT_PATH'] = args.checkpoint
    
    print("=" * 80)
    print("Class Agnostic Counting Web Visualization Application")
    print("=" * 80)
    print(f"Config file: {args.config}")
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Access URL: http://{args.host}:{args.port}")
    print("=" * 80)
    
    app.run(host=args.host, port=args.port, debug=args.debug)

