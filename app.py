from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

CLASSES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128*16*16,128), nn.ReLU(), nn.Linear(128,15))
    def forward(self,x):
        return self.fc(self.cnn(x))

model = CNN()
model.load_state_dict(torch.load('potato_model.pth', map_location='cpu'))
model.eval()
transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return "Please upload image from homepage", 200
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file'}), 400
    img = Image.open(file).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img).argmax(1).item()
    return jsonify({'prediction': CLASSES[pred]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
