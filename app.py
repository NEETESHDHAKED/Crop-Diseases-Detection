import torch, torch.nn as nn
from torchvision import transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os

app = Flask(__name__)

CLASSES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128*16,128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128,15))
    def forward(self,x):
        return self.fc(self.cnn(x))

model = CNN()
model.load_state_dict(torch.load('potato_model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out,1)[0]
        idx = prob.argmax().item()
    return jsonify({'prediction': CLASSES[idx], 'confidence': f"{prob[idx].item()*100:.2f}%"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
