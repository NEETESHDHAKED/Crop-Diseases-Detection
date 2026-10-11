import os
import numpy as np
from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model('model.h5')

classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

@app.route('/', methods=['GET'])
def home():
    return '''
    <html><body style="font-family:Arial; text-align:center; margin-top:50px;">
    <h1>Plant Disease Detection</h1>
    <form action="/" method="post" enctype="multipart/form-data">
    <input type="text" name="name" placeholder="Kisan ka naam" required><br><br>
    <input type="file" name="file" required><br><br>
    <input type="submit" value="Check Karo">
    </form></body></html>
    '''

@app.route('/', methods=['POST'])
def predict():
    try:
        f = request.files['file']
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, f.filename)
        f.save(path)

        img = image.load_img(path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        pred = model.predict(x)
        result = classes[np.argmax(pred)]
        return f"<h1 style='text-align:center; margin-top:50px;'>Result: {result}</h1><center><a href='/'>Wapas Jao</a></center>"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h1>Error: {str(e)}</h1><pre>{traceback.format_exc()}</pre>"

if __name__ == '__main__':
    app.run()
