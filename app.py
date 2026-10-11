import os
import numpy as np
from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Model ko global load
model = None
def get_model():
    global model
    if model is None:
        model = load_model('model.h5')
    return model

classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            m = get_model()
            f = request.files['file']
            upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            path = os.path.join(upload_dir, f.filename)
            f.save(path)

            img = image.load_img(path, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            pred = m.predict(x)
            result = classes[np.argmax(pred)]

            return f'''
            <html><head><style>
            body{{background: linear-gradient(135deg, #a8e063, #56ab2f); font-family:Arial; display:flex; justify-content:center; align-items:center; height:100vh; margin:0;}}
            .card{{background:white; padding:40px; border-radius:20px; box-shadow:0 10px 30px rgba(0,0,0,0.2); text-align:center;}}
            .result{{color:#2e7d32; font-size:22px; font-weight:bold;}}
            a{{text-decoration:none; background:#56ab2f; color:white; padding:10px 20px; border-radius:10px; display:inline-block; margin-top:20px;}}
            </style></head><body>
            <div class="card"><h2>🌿 Result</h2><p class="result">{result}</p><a href="/">Wapas Jao</a></div></body></html>
            '''
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"<h3>Error: {e}</h3><pre>{traceback.format_exc()}</pre>"

    return '''
    <html><head><style>
    body{background: linear-gradient(135deg, #e8f5e9, #a5d6a7, #66bb6a); font-family:Arial; display:flex; justify-content:center; align-items:center; height:100vh; margin:0;}
    .card{background:rgba(255,255,255,0.95); padding:35px; border-radius:20px; box-shadow:0 15px 35px rgba(0,0,0,0.2); text-align:center; width:350px;}
    h1{color:#2e7d32;} input[type="text"],input[type="file"]{width:90%; padding:12px; margin:10px 0; border:2px solid #a5d6a7; border-radius:10px;}
    input[type="submit"]{background:linear-gradient(to right, #43a047, #2e7d32); color:white; border:none; padding:12px 30px; border-radius:25px; cursor:pointer; font-size:16px; font-weight:bold; width:100%;}
    </style></head><body>
    <div class="card"><h1>🌱 Plant Disease Detection</h1><p>Khet ki fasal ka rog pehchano</p>
    <form method="post" enctype="multipart/form-data"><input type="text" name="name" placeholder="Kisan ka naam" required><br><input type="file" name="file" required><br><input type="submit" value="Check Karo 🌿"></form></div></body></html>
    '''

if __name__ == '__main__':
    app.run()
