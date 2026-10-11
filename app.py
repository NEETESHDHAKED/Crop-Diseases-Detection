import os
import numpy as np
from flask import Flask, request
import tensorflow as tf
import inspect

print("Applying FINAL compatibility patch...")

# --- UNIVERSAL PATCH ---
def make_universal_patcher(original_init):
    def universal_patched_init(self, *args, **kwargs):
        # Purane model ke sab faltu args hata do
        for bad_key in ['input_axes', 'output_axes', 'distribution', 'dtype', 'batch_shape', 'optional']:
            kwargs.pop(bad_key, None)
        try:
            return original_init(self, *args, **kwargs)
        except TypeError:
            # Agar phir bhi error aaye to sirf 'seed' rakho, baaki sab hatao
            safe_kwargs = {}
            if 'seed' in kwargs:
                safe_kwargs['seed'] = kwargs['seed']
            # signature check karke sirf allowed args rakho
            try:
                sig = inspect.signature(original_init)
                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                return original_init(self, *args, **filtered)
            except:
                return original_init(self, *args, **safe_kwargs)
    return universal_patched_init

# 1. InputLayer patch
try:
    from keras.engine.input_layer import InputLayer
    InputLayer.__init__ = make_universal_patcher(InputLayer.__init__)
    print("Keras InputLayer patched")
except: pass
try:
    from tensorflow.keras.layers import InputLayer as TFInputLayer
    TFInputLayer.__init__ = make_universal_patcher(TFInputLayer.__init__)
except: pass

# 2. ALL Initializers patch
patched_classes = []
try:
    import keras.initializers.initializers_v2 as kv2
    import keras.initializers.initializers as kold
    import tensorflow.python.keras.initializers.initializers_v2 as tfv2
    
    for module in [kv2, kold, tfv2]:
        for name in ['GlorotUniform', 'GlorotNormal', 'HeUniform', 'HeNormal', 'LecunUniform', 'LecunNormal', 'Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'Orthogonal', 'Identity']:
            if hasattr(module, name):
                cls = getattr(module, name)
                if cls not in patched_classes:
                    cls.__init__ = make_universal_patcher(cls.__init__)
                    patched_classes.append(cls)
    print(f"Patched {len(patched_classes)} initializers")
except Exception as e:
    print(f"Patch error: {e}")

from tensorflow.keras.preprocessing import image
# --- PATCH END ---

app = Flask(__name__)
model = None

def get_model():
    global model
    if model is None:
        print("Loading model.h5...")
        model = tf.keras.models.load_model('model.h5', compile=False)
        print("Model loaded OK!")
    return model

classes = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            m = get_model()
            f = request.files['file']
            name = request.form.get('name','Kisan')
            os.makedirs('uploads', exist_ok=True)
            path = os.path.join('uploads', f.filename)
            f.save(path)
            img = image.load_img(path, target_size=(64,64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            pred = m.predict(x)
            res = classes[np.argmax(pred)]
            clean = res.replace("___"," - ").replace("__"," ").replace("_"," ")
            conf = float(np.max(pred)*100)
            return f'''<html><head><meta name="viewport" content="width=device-width, initial-scale=1.0"><style>body{{background:linear-gradient(135deg,#a8e063 0%,#56ab2f 100%);font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}}.card{{background:white;padding:35px;border-radius:20px;box-shadow:0 15px 35px rgba(0,0,0,0.2);text-align:center;max-width:400px;width:90%}}.result{{color:#2e7d32;font-size:22px;font-weight:bold;background:#e8f5e9;padding:15px;border-radius:10px;margin:15px 0}}a{{text-decoration:none;background:#56ab2f;color:white;padding:12px 25px;border-radius:25px;display:inline-block;margin-top:15px;font-weight:bold}}</style></head><body><div class="card"><h2>🌿 Namaste {name}!</h2><div class="result">{clean}</div><p>Confidence: {conf:.2f}%</p><a href="/">🔄 Dusri Photo Check Karo</a></div></body></html>'''
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"<h2>Error: {e}</h2><pre>{traceback.format_exc()}</pre>"
    return '''<html><head><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Crop Disease Detection</title><style>body{background:linear-gradient(135deg,#e8f5e9 0%,#a5d6a7 50%,#66bb6a 100%);background-attachment:fixed;font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}.card{background:rgba(255,255,255,0.95);padding:35px;border-radius:20px;box-shadow:0 15px 35px rgba(0,0,0,0.2);text-align:center;width:350px;max-width:90%}h1{color:#2e7d32}input[type="text"],input[type="file"]{width:90%;padding:12px;margin:10px 0;border:2px solid #a5d6a7;border-radius:10px}input[type="submit"]{background:linear-gradient(to right,#43a047,#2e7d32);color:white;border:none;padding:13px 30px;border-radius:25px;cursor:pointer;font-size:16px;font-weight:bold;width:95%;margin-top:15px}</style></head><body><div class="card"><h1>🌱 Plant Disease Detection</h1><p>Khet ki fasal ka rog pehchano</p><form method="post" enctype="multipart/form-data"><input type="text" name="name" placeholder="Kisan ka naam" required><br><input type="file" name="file" accept="image/*" required><br><input type="submit" value="Check Karo 🌿"></form></div></body></html>'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
