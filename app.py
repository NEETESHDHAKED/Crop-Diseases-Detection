import os
import numpy as np
from flask import Flask, request
import tensorflow as tf

print("PATCHING - FINAL V3...")

# --- SUPER PATCH: from_config ko bhi patch karo ---
def patch_all():
    import keras.initializers
    from keras.initializers import initializers_v2
    from keras.engine.input_layer import InputLayer
    
    # 1. InputLayer
    orig_il = InputLayer.__init__
    def new_il(self, *a, **kw):
        kw.pop('batch_shape', None)
        kw.pop('batch_input_shape', None)
        if 'batch_shape' in kw: kw['batch_input_shape'] = kw.pop('batch_shape')
        # also handle if both missing, ignore
        kw.pop('batch_shape', None)
        kw.pop('optional', None)
        return orig_il(self, *a, **kw)
    InputLayer.__init__ = new_il

    # 2. Patch Initializer.from_config - YAHI ASLI FIX HAI
    orig_from_config = initializers_v2.Initializer.from_config
    @classmethod
    def new_from_config(cls, config):
        # Purane faltu keys hatao
        config = dict(config)  # copy
        config.pop('input_axes', None)
        config.pop('output_axes', None)
        config.pop('dtype', None)
        config.pop('distribution', None)
        return orig_from_config.__func__(cls, config)
    
    initializers_v2.Initializer.from_config = new_from_config
    
    # 3. Also patch __init__ of all initializers as backup
    for name in ['GlorotUniform','GlorotNormal','HeUniform','HeNormal','Zeros','Ones']:
        if hasattr(initializers_v2, name):
            Cls = getattr(initializers_v2, name)
            orig_init = Cls.__init__
            def make_new(orig):
                def new_init(self, *a, **kw):
                    kw.pop('input_axes', None)
                    kw.pop('output_axes', None)
                    kw.pop('dtype', None)
                    return orig(self, *a, **kw)
                return new_init
            Cls.__init__ = make_new(orig_init)
    print("PATCH DONE!")

patch_all()

from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = None
def get_model():
    global model
    if model is None:
        print("Loading model.h5...")
        model = tf.keras.models.load_model('model.h5', compile=False)
        print("LOADED!")
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