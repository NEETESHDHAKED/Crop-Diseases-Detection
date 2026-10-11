from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np

app = Flask(__name__)

CLASSES = [
 'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy'
]

device = torch.device("cpu")

# Simple CNN Model (tera wala)
class PlantModel(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self,x):
        return self.fc(self.cnn(x))

def load_model():
    ckpt_path = "potato_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Agar checkpoint dict ke andar hai
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        checkpoint = checkpoint['model_state']

    # Check karo ResNet hai kya
    is_resnet = any('layer1' in k for k in checkpoint.keys()) if isinstance(checkpoint, dict) else False

    if is_resnet:
        print("ResNet model detected")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        model.load_state_dict(checkpoint, strict=False)
    else:
        # Agar pura model save kiya hai (not state_dict)
        if not isinstance(checkpoint, dict) or not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            try:
                # Direct model object
                model = torch.load(ckpt_path, map_location=device)
                if isinstance(model, nn.Module):
                    return model.eval()
            except:
                pass

        print("Simple CNN model detected")
        model = PlantModel(num_classes=len(CLASSES))
        try:
            model.load_state_dict(checkpoint, strict=True)
        except Exception as e:
            print(f"Strict load failed: {e}, trying non-strict")
            model.load_state_dict(checkpoint, strict=False)

    return model.eval()

model = load_model()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')

        np_img = np.array(img.resize((100,100)))
        green_ratio = np.mean((np_img[:,:,1] > 60) & (np_img[:,:,1] > np_img[:,:,0]))

        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, 1)[0]
            idx = prob.argmax().item()
            pred = CLASSES[idx]
            conf = prob[idx].item()*100

        if green_ratio < 0.08:
            return jsonify({'prediction': '⚠️ Ye patte ki photo nahi hai! Patte ki photo daalein', 'confidence': '0%'})

        if conf < 50:
            return jsonify({'prediction': 'Photo clear nahi hai, saaf patte ki photo daalein', 'confidence': f"{conf:.2f}%"})

        clean_pred = pred.replace('___',' - ').replace('__',' ').replace('_',' ')
        return jsonify({'prediction': clean_pred, 'confidence': f"{conf:.2f}%", 'status':'ok'})
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({'prediction': f"Error: {str(e)}", 'confidence': "0%"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
