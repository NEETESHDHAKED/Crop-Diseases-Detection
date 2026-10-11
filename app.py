from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
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
            nn.Linear(128*4*4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self,x):
        return self.fc(self.cnn(x))

device = torch.device("cpu")
model = PlantModel(num_classes=len(CLASSES))
model.load_state_dict(torch.load("potato_model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((32,32)),
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
            return jsonify({'prediction': 'Photo clear nahi hai', 'confidence': f"{conf:.2f}%"})

        clean_pred = pred.replace('___',' - ').replace('__',' ').replace('_',' ')
        return jsonify({'prediction': clean_pred, 'confidence': f"{conf:.2f}%"})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'prediction': f"Error: {str(e)}", 'confidence': "0%"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
