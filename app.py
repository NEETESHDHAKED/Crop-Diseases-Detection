import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from flask import Flask, request, jsonify, render_template
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import math

app = Flask(__name__)

CLASSES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

ckpt = torch.load('potato_model.pth', map_location='cpu')
sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

fc_in = None
for k,v in sd.items():
    if isinstance(v, torch.Tensor) and v.dim()==2 and v.shape[0]==128:
        fc_in = v.shape[1]
        break

pool_size = int(math.sqrt(fc_in // 128)) if fc_in else 16

class CNN(nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((pool,pool))
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128*pool*pool,128), nn.ReLU(), nn.Linear(128,15))
    def forward(self,x):
        return self.fc(self.cnn(x))

model = CNN(pool_size)
model.load_state_dict(sd, strict=False)
model.eval()

transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        img = Image.open(file.stream).convert('RGB')
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = model(img).argmax(1).item()
        return jsonify({'prediction': CLASSES[pred]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
