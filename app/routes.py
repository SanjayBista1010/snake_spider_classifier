import os
import torch
from flask import render_template, request, jsonify
from PIL import Image
import io
import base64

from app import app

def predict_image(image, model, device, transform, class_names):
    """Make prediction on a single image"""
    try:
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        # Convert to percentages
        confidence_percent = confidence.item() * 100
        class_probs = {
            class_names[i]: {
                'percent': f"{probs[0][i].item()*100:.2f}%",
                'value': probs[0][i].item()
            } for i in range(len(class_names))
        }
        
        # Get class with color
        class_colors = {
            "snake": "success",
            "spider": "danger"
        }
        
        return {
            'prediction': class_names[pred_idx.item()],
            'confidence': f"{confidence_percent:.2f}%",
            'confidence_value': confidence_percent,
            'all_probabilities': class_probs,
            'class_color': class_colors.get(class_names[pred_idx.item()], "primary"),
            'success': True
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    model_info = {
        'loaded': app.config.get('model') is not None,
        'device': str(app.config.get('device')),
        'class_names': app.config.get('class_names', [])
    }
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image.'})
    
    try:
        # Read and validate image
        image = Image.open(file.stream).convert('RGB')
        
        # Get model and config
        model = app.config.get('model')
        device = app.config.get('device')
        transform = app.config.get('transform')
        class_names = app.config.get('class_names')
        
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded. Please check if the model file exists.'})
        
        # Make prediction
        result = predict_image(image, model, device, transform, class_names)
        
        if result['success']:
            # Convert image to base64 for display
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            result['image_data'] = f"data:image/jpeg;base64,{img_str}"
            result['filename'] = file.filename
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle multiple image predictions"""
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'})
    
    files = request.files.getlist('files')
    valid_files = [f for f in files if f.filename != '']
    
    if not valid_files:
        return jsonify({'success': False, 'error': 'No valid files selected'})
    
    # Limit batch size
    if len(valid_files) > 10:
        return jsonify({'success': False, 'error': 'Maximum 10 files allowed per batch'})
    
    try:
        model = app.config.get('model')
        device = app.config.get('device')
        transform = app.config.get('transform')
        class_names = app.config.get('class_names')
        
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
        results = []
        for file in valid_files:
            try:
                image = Image.open(file.stream).convert('RGB')
                result = predict_image(image, model, device, transform, class_names)
                
                if result['success']:
                    # Convert image to base64 for display
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    result['image_data'] = f"data:image/jpeg;base64,{img_str}"
                    result['filename'] = file.filename
                    results.append(result)
                else:
                    results.append({
                        'success': False,
                        'error': result['error'],
                        'filename': file.filename
                    })
                    
            except Exception as e:
                results.append({
                    'success': False,
                    'error': f'Error processing {file.filename}: {str(e)}',
                    'filename': file.filename
                })
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing images: {str(e)}'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_loaded = app.config.get('model') is not None
    model_info = {}
    
    if model_loaded:
        model = app.config.get('model')
        model_info = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'device': str(next(model.parameters()).device)
        }
    
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded,
        'device': str(app.config.get('device')),
        'class_names': app.config.get('class_names', []),
        'model_info': model_info
    })

@app.route('/model_info')
def model_info():
    """Detailed model information"""
    model = app.config.get('model')
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    info = {
        'model_type': 'CustomVGG16 with SE Blocks',
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': str(next(model.parameters()).device),
        'class_names': app.config.get('class_names', []),
        'input_size': '224x224',
        'architecture': 'VGG16 + Squeeze-Excitation Blocks'
    }
    
    return jsonify(info)