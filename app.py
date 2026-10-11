import os, json, h5py
import numpy as np
from flask import Flask, request
import tensorflow as tf

print("ULTIMATE PATCH V8 - All in one...")

# --- PATCH A: InputLayer - batch_shape, optional ---
try:
    from keras.engine.input_layer import InputLayer as K_Input
    orig_k = K_Input.__init__
    def new_k(self, *a, **kw):
        if 'batch_shape' in kw:
            if 'batch_input_shape' not in kw:
                kw['batch_input_shape'] = kw.pop('batch_shape')
            else:
                kw.pop('batch_shape')
        kw.pop('optional', None)
        kw.pop('batch_shape', None)
        kw.pop('sparse', None)
        kw.pop('ragged', None)
        return orig_k(self, *a, **kw)
    K_Input.__init__ = new_k
    print("Keras InputLayer patched")
except Exception as e:
    print(f"K Input fail: {e}")

try:
    import keras.layers as KL
    if hasattr(KL, 'InputLayer'):
        orig = KL.InputLayer.__init__
        def new_kl(self, *a, **kw):
            if 'batch_shape' in kw:
                if 'batch_input_shape' not in kw:
                    kw['batch_input_shape'] = kw.pop('batch_shape')
                else:
                    kw.pop('batch_shape')
            kw.pop('optional', None)
            kw.pop('batch_shape', None)
            return orig(self, *a, **kw)
        KL.InputLayer.__init__ = new_kl
        print("KL InputLayer patched")
except: pass

# --- PATCH B: validate_kwargs - quantization_config ---
import keras.utils.generic_utils as gu
orig_validate = gu.validate_kwargs
def universal_validate(kwargs, allowed_kwargs, error_message="Keyword not understood:"):
    bad = ['quantization_config','quantization','dtype_policy','DTypePolicy','dtype_policies',
           'input_axes','output_axes','distribution','synchronization','aggregation',
           'trainable','autocast','input_dtype','module','batch_shape','optional']
    for k in bad:
        kwargs.pop(k, None)
    cleaned = {k:v for k,v in kwargs.items() if k in allowed_kwargs}
    return orig_validate(cleaned, allowed_kwargs, error_message)
gu.validate_kwargs = universal_validate
print("validate_kwargs patched")

# --- PATCH C: DTypePolicy ---
orig_cc = gu.class_and_config_for_serialized_keras_object
def patched_cc(config, *a, **kw):
    if isinstance(config, dict) and config.get('class_name') == 'DTypePolicy':
        return (tf.keras.mixed_precision.Policy, {'name': 'float32'})
    return orig_cc(config, *a, **kw)
gu.class_and_config_for_serialized_keras_object = patched_cc
print("DTypePolicy patched")

from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = None

def deep_clean(obj):
    if isinstance(obj, dict):
        for k in ['quantization_config','quantization','dtype_policy','DTypePolicy','_quantization_config','module','optional']:
            obj.pop(k, None)
        # batch_shape ko batch_input_shape me convert
        if 'batch_shape' in obj and 'batch_input_shape' not in obj:
            obj['batch_input_shape'] = obj.pop('batch_shape')
        if obj.get('class_name') == 'DTypePolicy':
            obj['class_name'] = 'Policy'
            obj['config'] = {'name': 'float32'}
        for v in obj.values():
            deep_clean(v)
    elif isinstance(obj, list):
        for item in obj:
            deep_clean(item)

def get_model():
    global model
    if model is None:
        print("Reading model.h5...")
        try:
            with h5py.File('model.h5', 'r') as f:
                mc = f.attrs.get('model_config')
                if isinstance(mc, bytes): mc = mc.decode('utf-8')
                cfg = json.loads(mc)
                deep_clean(cfg)
                j = json.dumps(cfg)
                print("Cleaned, creating model...")
                co = {'DTypePolicy': tf.keras.mixed_precision.Policy('float32'), 'Policy': tf.keras.mixed_precision.Policy}
                model = tf.keras.models.model_from_json(j, custom_objects=co)
                print("Loading weights...")
                model.load_weights('model.h5')
                print("MODEL LOADED SUCCESSFULLY!")
        except Exception as e:
            print(f"JSON method failed: {e}")
            import traceback; traceback.print_exc()
            # fallback direct
            co = {'DTypePolicy': tf.keras.mixed_precision.Policy('float32'), 'Policy': tf.keras.mixed_precision.Policy}
            model = tf.keras.models.load_model('model.h5', compile=False, custom_objects=co)
            print("MODEL LOADED DIRECT!")
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
            import traceback; traceback.print_exc()
            return f"<h3>Error: {e}</h3><pre>{traceback.format_exc()}</pre>"
    return '''<html><head><meta name="viewport" content="width=device-width"><title>Crop Disease</title><style>body{background:linear-gradient(135deg,#e8f5e9 0%,#a5d6a7 50%,#66bb6a 100%);font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}.card{background:rgba(255,255,255,0.95);padding:35px;border-radius:20px;text-align:center;width:350px;max-width:90%}h1{color:#2e7d32}input[type="text"],input[type="file"]{width:90%;padding:12px;margin:10px 0;border:2px solid #a5d6a7;border-radius:10px}input[type="submit"]{background:linear-gradient(to right,#43a047,#2e7d32);color:white;border:none;padding:13px 30px;border-radius:25px;cursor:pointer;font-size:16px;font-weight:bold;width:95%;margin-top:15px}</style></head><body><div class="card"><h1>🌱 Plant Disease Detection</h1><p>Khet ki fasal ka rog pehchano</p><form method="post" enctype="multipart/form-data"><input type="text" name="name" placeholder="Kisan ka naam" required><br><input type="file" name="file" accept="image/*" required><br><input type="submit" value="Check Karo 🌿"></form></div></body></html>'''

if __name__=='__main__':
    app.run(host='0.0.0.0',port=10000)
