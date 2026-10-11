import os, json, h5py
import numpy as np
from flask import Flask, request
import tensorflow as tf

print("ULTIMATE V9 - Fixing ALL...")

# --- 1. InputLayer batch_shape, optional ---
try:
    from keras.engine.input_layer import InputLayer
    o = InputLayer.__init__
    def n(self,*a,**kw):
        if 'batch_shape' in kw:
            if 'batch_input_shape' not in kw:
                kw['batch_input_shape']=kw.pop('batch_shape')
            else: kw.pop('batch_shape')
        kw.pop('optional',None); kw.pop('batch_shape',None)
        return o(self,*a,**kw)
    InputLayer.__init__=n
    print("InputLayer patched")
except Exception as e: print(e)

# --- 2. GlorotUniform input_axes, output_axes - YEHI TERA CURRENT ERROR HAI ---
try:
    from keras.initializers import initializers_v2
    orig_fc = initializers_v2.Initializer.from_config
    @classmethod
    def new_fc(cls, config):
        config=dict(config)
        for k in ['input_axes','output_axes','dtype','distribution','quantization_config','dtype_policy','DTypePolicy']:
            config.pop(k,None)
        return orig_fc.__func__(cls, config)
    initializers_v2.Initializer.from_config=new_fc
    print("Initializer from_config patched")

    # Backup: __init__ bhi patch
    for name in ['GlorotUniform','GlorotNormal','HeUniform','HeNormal']:
        if hasattr(initializers_v2, name):
            Cls=getattr(initializers_v2, name)
            oi=Cls.__init__
            def mk(o):
                def ni(self,*a,**kw):
                    kw.pop('input_axes',None); kw.pop('output_axes',None); kw.pop('dtype',None)
                    return o(self,*a,**kw)
                return ni
            Cls.__init__=mk(oi)
    print("Glorot patched")
except Exception as e: print(f"Init patch fail {e}")

# --- 3. validate_kwargs quantization_config ---
import keras.utils.generic_utils as gu
ov=gu.validate_kwargs
def uv(kwargs, allowed, msg="Keyword not understood:"):
    for k in ['quantization_config','quantization','dtype_policy','DTypePolicy','input_axes','output_axes','batch_shape','optional']:
        kwargs.pop(k,None)
    cleaned={k:v for k,v in kwargs.items() if k in allowed}
    return ov(cleaned, allowed, msg)
gu.validate_kwargs=uv
print("validate_kwargs patched")

# --- 4. DTypePolicy ---
oc=gu.class_and_config_for_serialized_keras_object
def pc(config,*a,**kw):
    if isinstance(config,dict) and config.get('class_name')=='DTypePolicy':
        return (tf.keras.mixed_precision.Policy,{'name':'float32'})
    return oc(config,*a,**kw)
gu.class_and_config_for_serialized_keras_object=pc
print("DTypePolicy patched")

from tensorflow.keras.preprocessing import image
app=Flask(__name__)
model=None

def deep_clean(o):
    if isinstance(o,dict):
        for k in ['quantization_config','dtype_policy','DTypePolicy','optional']:
            o.pop(k,None)
        if 'batch_shape' in o and 'batch_input_shape' not in o:
            o['batch_input_shape']=o.pop('batch_shape')
        if o.get('class_name')=='DTypePolicy':
            o['class_name']='Policy'; o['config']={'name':'float32'}
        for v in o.values(): deep_clean(v)
    elif isinstance(o,list):
        for i in o: deep_clean(i)

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('model_fixed.h5', compile=False)
    return model

classes=['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        try:
            m=get_model()
            f=request.files['file']; name=request.form.get('name','Kisan')
            os.makedirs('uploads',exist_ok=True); p=os.path.join('uploads',f.filename); f.save(p)
            img=image.load_img(p,target_size=(64,64)); x=image.img_to_array(img); x=np.expand_dims(x,axis=0)/255.0
            pred=m.predict(x); res=classes[np.argmax(pred)]; clean=res.replace("___"," - ").replace("__"," ").replace("_"," "); conf=float(np.max(pred)*100)
            return f'<html><head><meta name="viewport" content="width=device-width"><style>body{{background:linear-gradient(135deg,#a8e063 0%,#56ab2f 100%);font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}}.card{{background:white;padding:35px;border-radius:20px;text-align:center;max-width:400px;width:90%}}.result{{color:#2e7d32;font-size:22px;font-weight:bold;background:#e8f5e9;padding:15px;border-radius:10px;margin:15px 0}}a{{text-decoration:none;background:#56ab2f;color:white;padding:12px 25px;border-radius:25px;display:inline-block;margin-top:15px}}</style></head><body><div class="card"><h2>🌿 Namaste {name}!</h2><div class="result">{clean}</div><p>Confidence: {conf:.2f}%</p><a href="/">🔄 Dusri Photo</a></div></body></html>'
        except Exception as e:
            import traceback; traceback.print_exc()
            return f"<h3>Error: {e}</h3><pre>{traceback.format_exc()}</pre>"
    return '''<html><head><meta name="viewport" content="width=device-width"><title>Crop Disease</title><style>body{background:linear-gradient(135deg,#e8f5e9 0%,#a5d6a7 50%,#66bb6a 100%);font-family:Arial;display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0}.card{background:rgba(255,255,255,0.95);padding:35px;border-radius:20px;text-align:center;width:350px;max-width:90%}h1{color:#2e7d32}input[type="text"],input[type="file"]{width:90%;padding:12px;margin:10px 0;border:2px solid #a5d6a7;border-radius:10px}input[type="submit"]{background:linear-gradient(to right,#43a047,#2e7d32);color:white;border:none;padding:13px 30px;border-radius:25px;cursor:pointer;font-size:16px;font-weight:bold;width:95%;margin-top:15px}</style></head><body><div class="card"><h1>🌱 Plant Disease Detection</h1><p>Khet ki fasal ka rog pehchano</p><form method="post" enctype="multipart/form-data"><input type="text" name="name" placeholder="Kisan ka naam" required><br><input type="file" name="file" accept="image/*" required><br><input type="submit" value="Check Karo 🌿"></form></div></body></html>'''

if __name__=='__main__': app.run(host='0.0.0.0',port=10000)
