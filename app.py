import os
import numpy as np
from flask import Flask, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = None

def get_model():
    global model
    if model is None:
        # model_fixed.h5 already legacy format hai, direct load hoga
        model = tf.keras.models.load_model('model_fixed.h5', compile=False)
        print("Model loaded:", model.output_shape)
    return model

# 15 naam hai par model 16 output de raha hai -> 16va ignore karenge
classes=['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        m=get_model()
        f=request.files['file']
        name=request.form.get('name','Kisan')
        os.makedirs('uploads',exist_ok=True)
        p=os.path.join('uploads',f.filename)
        f.save(p)

        img=image.load_img(p,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)/255.0

        pred_full = m.predict(x)[0] # 16 values
        pred = pred_full[:15] # sirf pehle 15 liye, 16va hata diya
        idx = np.argmax(pred)
        res = classes[idx]
        clean=res.replace("___"," - ").replace("__"," ").replace("_"," ")
        conf=float(np.max(pred)*100)

        return f'<html><head><meta name="viewport" content="width=device-width"><style>body{{background:linear-gradient(135deg,#a8e063 0%,#56ab2f 100%);font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}}.card{{background:white;padding:35px;border-radius:20px;text-align:center;max-width:400px;width:90%}}.result{{color:#2e7d32;font-size:22px;font-weight:bold;background:#e8f5e9;padding:15px;border-radius:10px;margin:15px 0}}a{{text-decoration:none;background:#56ab2f;color:white;padding:12px 25px;border-radius:25px;display:inline-block;margin-top:15px}}</style></head><body><div class="card"><h2>🌿 Namaste {name}!</h2><div class="result">{clean}</div><p>Confidence: {conf:.2f}%</p><p style="font-size:12px;color:gray">Debug: Full pred {pred_full}</p><a href="/">🔄 Dusri Photo</a></div></body></html>'
    return '''<html><head><meta name="viewport" content="width=device-width"><title>Crop Disease</title><style>body{background:linear-gradient(135deg,#e8f5e9 0%,#a5d6a7 50%,#66bb6a 100%);font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}.card{background:rgba(255,255,255,0.95);padding:35px;border-radius:20px;text-align:center;width:350px;max-width:90%}h1{color:#2e7d32}input[type="text"],input[type="file"]{width:90%;padding:12px;margin:10px 0;border:2px solid #a5d6a7;border-radius:10px}input[type="submit"]{background:linear-gradient(to right,#43a047,#2e7d32);color:white;border:none;padding:13px 30px;border-radius:25px;cursor:pointer;font-size:16px;font-weight:bold;width:95%;margin-top:15px}</style></head><body><div class="card"><h1>🌱 Plant Disease Detection</h1><p>Khet ki fasal ka rog pehchano</p><form method="post" enctype="multipart/form-data"><input type="text" name="name" placeholder="Kisan ka naam" required><br><input type="file" name="file" accept="image/*" required><br><input type="submit" value="Check Karo 🌿"></form></div></body></html>'''

if __name__=='__main__':
    app.run(host='0.0.0.0',port=10000)
