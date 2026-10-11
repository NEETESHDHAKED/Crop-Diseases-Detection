import os
import numpy as np
from flask import Flask, request
import tensorflow as tf

print("PATCHING V4 - DTypePolicy + GlorotUniform...")

# --- PATCH 1: GlorotUniform input_axes/output_axes ---
def patch_initializers():
    try:
        from keras.initializers import initializers_v2
        from keras.engine.input_layer import InputLayer
        orig_from_config = initializers_v2.Initializer.from_config
        @classmethod
        def new_from_config(cls, config):
            config = dict(config)
            for k in ['input_axes','output_axes','dtype','distribution']:
                config.pop(k, None)
            return orig_from_config.__func__(cls, config)
        initializers_v2.Initializer.from_config = new_from_config
        print("Initializer patched")
        
        orig_il = InputLayer.__init__
        def new_il(self, *a, **kw):
            kw.pop('batch_shape', None)
            kw.pop('batch_input_shape', None)
            kw.pop('optional', None)
            return orig_il(self, *a, **kw)
        InputLayer.__init__ = new_il
    except Exception as e:
        print(f"Init patch fail: {e}")

patch_initializers()

# --- PATCH 2: DTypePolicy ---
# TF 2.10 me DTypePolicy nahi hai, to usko fake banao
try:
    import keras.utils.generic_utils as generic_utils
    orig_class_and_config = generic_utils.class_and_config_for_serialized_keras_object
    def patched_class_and_config(config, *args, **kwargs):
        if isinstance(config, dict) and config.get('class_name') == 'DTypePolicy':
            print("Intercepted DTypePolicy -> returning float32 Policy")
            # float32 policy return karo
            return (tf.keras.mixed_precision.Policy, {'name': 'float32'})
        return orig_class_and_config(config, *args, **kwargs)
    generic_utils.class_and_config_for_serialized_keras_object = patched_class_and_config
    print("DTypePolicy patched")
except Exception as e:
    print(f"DTypePolicy patch fail: {e}")

from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = None

def get_model():
    global model
    if model is None:
        print("Loading model.h5 with custom_objects...")
        # custom_objects me bhi DTypePolicy daal do - double safety
        custom_objects = {
            'DTypePolicy': tf.keras.mixed_precision.Policy('float32'),
            'Policy': tf.keras.mixed_precision.Policy,
        }
        model = tf.keras.models.load_model('model.h5', compile=False, custom_objects=custom_objects)
        print("MODEL LOADED SUCCESSFULLY!")
    return model

classes = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        try:
            m=get_model()
            f=request.files['file']
            name=request.form.get('name','Kisan')
            os.makedirs('uploads',exist_ok=True)
            p=os.path.join('uploads',f.filename)
            f.save(p)
            img=image.load_img(p,target_size=(64,64))
            x=image.img_to_array(img)
            x=np.expand_dims(x,axis=0)/255.0
            pred=m.predict(x)
            res=classes[np.argmax(pred)]
            clean=res.replace("___"," - ").replace("__"," ").replace("_"," ")
            conf=float(np.max(pred)*100)
            return f'<html><head><meta name="viewport" content="width=device-width"><style>body{{background:linear-gradient(135deg,#a8e063 0%,#56ab2f 100%);font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}}.card{{background:white;padding:35px;border-radius:20px;text-align:center;max-width:400px;width:90%}}.result{{color:#2e7d32;font-size:22px;font-weight:bold;background:#e8f5e9;padding:15px;border-radius:10px;margin:15px 0}}a{{text-decoration:none;background:#56ab2f;color:white;padding:12px 25px;border-radius:25px;display:inline-block;margin-top:15px}}</style></head><body><div class="card"><h2>🌿 Namaste {name}!</h2><div class="result">{clean}</div><p>Confidence: {conf:.2f}%</p><a href="/">🔄 Dusri Photo</a></div></body></html>'
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"<h3>Error: {e}</h3><pre>{traceback.format_exc()}</pre>"
    return '''<html><head><meta name="viewport" content="width=device-width"><title>Crop Disease</title><style>body{background:linear-gradient(135deg,#e8f5e9 0%,#a5d6a7 50%,#66bb6a 100%);font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}.card{background:rgba(255,255,255,0.95);padding:35px;border-radius:20px;text-align:center;width:350px;max-width:90%}h1{color:#2e7d32}input[type="text"],input[type="file"]{width:90%;padding:12px;margin:10px 0;border:2px solid #a5d6a7;border-radius:10px}input[type="submit"]{background:linear-gradient(to right,#43a047,#2e7d32);color:white;border:none;padding:13px 30px;border-radius:25px;cursor:pointer;font-size:16px;font-weight:bold;width:95%;margin-top:15px}</style></head><body><div class="card"><h1>🌱 Plant Disease Detection</h1><p>Khet ki fasal ka rog pehchano</p><form method="post" enctype="multipart/form-data"><input type="text" name="name" placeholder="Kisan ka naam" required><br><input type="file" name="file" accept="image/*" required><br><input type="submit" value="Check Karo 🌿"></form></div></body></html>'''

if __name__=='__main__':
    app.run(host='0.0.0.0',port=10000)
