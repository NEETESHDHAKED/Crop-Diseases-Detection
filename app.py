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
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')

        # 1. Hara rang check - patti me hara hona chahiye
        import numpy as np
        np_img = np.array(img.resize((100,100)))
        green_ratio = np.mean((np_img[:,:,1] > 80) & (np_img[:,:,1] > np_img[:,:,0]) & (np_img[:,:,1] > np_img[:,:,2]))

        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, 1)[0]
            idx = prob.argmax().item()
            pred = CLASSES[idx]
            conf = prob[idx].item()*100

        # 2. Agar hara kam hai to patti nahi hai
        if green_ratio < 0.10: # 10% se kam hara
            return jsonify({'prediction': '⚠️ Ye patti ki photo nahi hai! Kripya patte ki photo upload karein', 'confidence': '0%'})

        if conf < 60:
            return jsonify({'prediction': 'Photo clear nahi hai, saaf patte ki photo daalein', 'confidence': f"{conf:.2f}%"})

        if "Late_blight" in pred:
            final_pred = "Late Blight (Potato/Tomato) - Fungal disease"
        elif "Early_blight" in pred:
            final_pred = "Early Blight (Potato/Tomato)"
        else:
            final_pred = pred

        return jsonify({'prediction': final_pred, 'confidence': f"{conf:.2f}%"})
    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}", 'confidence': "0%"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
