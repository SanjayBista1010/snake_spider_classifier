import os
import sys
import torch
from PIL import Image
from flask import Flask
from torch.optim.swa_utils import AveragedModel

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.routes import app as flask_app
from models.vgg16_se import CustomVGG16
from snake_spider_classifier.features import transform_val

def create_app():
    """Create and configure Flask app with PROPER SWA model loading"""
    app = flask_app
    
    # Configure template and static folders
    app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'templates')
    app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'static')
    
    print(f"📁 Template folder: {app.template_folder}")
    print(f"📁 Static folder: {app.static_folder}")
    
    # Load model once at startup
    with app.app_context():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app.config['device'] = device
        
        # Load model
        model = CustomVGG16(num_classes=2).to(device)
        checkpoint_path = "reports/vgg16_se_swa_final.pth"
        
        if os.path.exists(checkpoint_path):
            print("🚀 Loading trained SWA model...")
            
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                print(f"📦 Checkpoint keys: {list(checkpoint.keys())}")
                
                # METHOD 1: Try to load as SWA model first
                if 'n_averaged' in checkpoint:
                    print("🎯 Detected SWA model - using AveragedModel approach")
                    
                    # Create SWA model wrapper
                    swa_model = AveragedModel(model)
                    
                    # Load SWA state
                    swa_model.load_state_dict(checkpoint)
                    
                    # Get the underlying model
                    model = swa_model.module
                    print("✅ SWA model loaded via AveragedModel wrapper")
                    
                else:
                    # METHOD 2: Direct state dict loading
                    print("🔧 Using direct state dict loading")
                    
                    # Get state dict (handle different key names)
                    if 'model_state' in checkpoint:
                        state_dict = checkpoint['model_state']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Remove 'module.' prefix if present
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('module.'):
                            new_key = key[7:]  # Remove 'module.' prefix
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    
                    # Load with strict=False to handle any mismatches
                    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                    
                    if missing_keys:
                        print(f"⚠️  Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                
                model.eval()
                print("✅ Model loaded successfully!")
                
                # Test the model with a dummy input to verify it's working
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224).to(device)
                    output = model(dummy_input)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    
                    print(f"🧪 Model test - Output shape: {output.shape}")
                    print(f"🧪 Model test - Probabilities: {probabilities.cpu().numpy()}")
                    
                    # Check if outputs are meaningful (not random)
                    max_prob = torch.max(probabilities).item()
                    if max_prob > 0.9:
                        print("🎯 Model test: EXCELLENT - High confidence outputs")
                    elif max_prob > 0.7:
                        print("✅ Model test: GOOD - Reasonable confidence outputs")
                    elif max_prob > 0.6:
                        print("⚠️  Model test: WEAK - Low confidence outputs")
                    else:
                        print("❌ Model test: POOR - Random-like outputs")
                
                # Print model info
                total_params = sum(p.numel() for p in model.parameters())
                print(f"📊 Model parameters: {total_params:,}")
                
                if 'best_acc' in checkpoint:
                    print(f"🏆 Best validation accuracy: {checkpoint['best_acc']:.2f}%")
                    
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()
                model = None
                
        else:
            print(f"❌ Checkpoint not found at: {checkpoint_path}")
            model = None
        
        app.config['model'] = model
        app.config['class_names'] = ["snake", "spider"]
        app.config['transform'] = transform_val
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    if app.config['model'] is not None:
        print("\n🌐 Starting Flask server...")
        print("   Access the app at: http://localhost:5000")
        print("   Press Ctrl+C to stop the server\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Cannot start server: Model failed to load")
        print("   Please check the error messages above and ensure your model file is valid")