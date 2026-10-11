import os
import numpy as np
from flask import Flask, request
import tensorflow as tf

# --- FULL COMPATIBILITY PATCH FOR OLD MODEL.H5 ---
print("Patching old TF model compatibility...")

# 1. InputLayer patch
try:
    from keras.engine.input_layer import InputLayer as KIL
    _orig = KIL.__init__
    def _patched_il(self, *a, **kw):
        kw.pop('batch_shape', None)
        kw.pop('optional', None)
        if 'batch_shape' in kw: kw['batch_input_shape'] = kw.pop('batch_shape')
        # agar batch_input_shape bhi nahi hai to batch_shape se banao
        if 'batch_input_shape' not in kw and 'batch_shape' in kw:
            kw['batch_input_shape'] = kw.pop('batch_shape')
        return _orig(self, *a, **kw)
    KIL.__init__ = _patched_il
except: pass

try:
    from tensorflow.keras.layers import InputLayer as TFIL
    _orig2 = TFIL.__init__
    def _patched_il2(self, *a, **kw):
        kw.pop('batch_shape', None)
        kw.pop('optional', None)
        kw.pop('batch_shape', None)
        # handle batch_shape
        if 'batch_shape' in kw:
            kw['batch_input_shape'] = kw.pop('batch_shape')
        return _orig2(self, *a, **kw)
    TFIL.__init__ = _patched_il2
except: pass

# 2. GlorotUniform and all initializers patch - input_axes hata do
def patch_initializer(cls):
    try:
        orig = cls.__init__
        def patched(self, *a, **kw):
            kw.pop('input_axes', None)
            kw.pop('dtype', None) # kabhi kabhi ye bhi problem karta hai
            return orig(self, *a, **kw)
        cls.__init__ = patched
    except: pass

try:
    import keras.initializers.initializers_v2 as init_v2
    for name in ['GlorotUniform', 'GlorotNormal', 'HeUniform', 'HeNormal', 'LecunUniform', 'LecunNormal', 'Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform']:
        if hasattr(init_v2, name):
            patch_initializer(getattr(init_v2, name))
    print("Initializers Patched!")
except Exception as e:
    print(f"Init patch warn: {e}")

from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = None

def get_model():
    global model
    if model is None:
        print("Loading Model...")
        # custom_objects ke saath load karo
        model = tf.keras.models.load_model('model.h5', compile=False, custom_objects={})
        print("Model Loaded Successfully!")
    return model

classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            m = get_model()
            f = request.files['file']
            farmer_name = request.form.get('name', 'Kisan')
            upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            path = os.path.join(upload_dir, f.filename)
            f.save(path)
            img = image.load_img(path, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            pred = m.predict(x)
            result = classes[np.argmax(pred)]
            clean_result = result.replace("___", " - ").replace("__", " ").replace("_", " ")
            confidence = float(np.max(pred) * 100)
            return f'''
            <html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
            body{{background: linear-gradient(135deg, #a8e063 0%, #56ab2f 100%); font-family: Arial; display:flex; justify-content:center; align-items:center; min-height:100vh; margin:0;}}
            .card{{background:white; padding:35px; border-radius:20px; box-shadow:0 15px 35px rgba(0,0,0,0.2); text-align:center; max-width:400px; width:90%;}}
            .result{{color:#2e7d32; font-size:22px; font-weight:bold; background:#e8f5e9; padding:15px; border-radius:10px; margin:15px 0;}}
            a{{text-decoration:none; background:#56ab2f; color:white; padding:12px 25px; border-radius:25px; display:inline-block; margin-top:15px; font-weight:bold;}}
            </style></head><body>
            <div class="card"><h2>🌿 Namaste {farmer_name}!</h2><p>Result:</p><div class="result">{clean_result}</div><p>Confidence: {confidence:.2f}%</p><a href="/">🔄 Dusri Photo Check Karo</a></div></body></html>'''
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"<h3>Error: {e}</h3><pre>{traceback.format_exc()}</pre>"

    return '''
    <html><head><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Crop Disease Detection</title>
    <style>
    body{background: linear-gradient(135deg, #e8f5e9 0%, #a5d6a7 50%, #66bb6a 100%); background-attachment: fixed; font-family: Arial; display:flex; justify-content:center; align-items:center; min-height:100vh; margin:0;}
    .card{background: rgba(255,255,255,0.95); padding:35px; border-radius:20px; box-shadow:0 15px 35px rgba(0,0,0,0.2); text-align:center; width:350px; max-width:90%;}
    h1{color:#2e7d32; margin-bottom:5px;} p{color:#666;}
    input[type="text"], input[type="file"]{width:90%; padding:12px; margin:10px 0; border:2px solid #a5d6a7; border-radius:10px; outline:none;}
    input[type="submit"]{background: linear-gradient(to right, #43a047, #2e7d32); color:white; border:none; padding:13px 30px; border-radius:25px; cursor:pointer; font-size:16px; font-weight:bold; width:95%; margin-top:15px;}
    </style></head><body>
    <div class="card"><h1>🌱 Plant Disease Detection</h1><p>Khet ki fasal ka rog pehchano</p><form method="post" enctype="multipart/form-data"><input type="text" name="name" placeholder="Kisan ka naam" required><br><input type="file" name="file" accept="image/*" required><br><input type="submit" value="Check Karo 🌿"></form></div></body></html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
