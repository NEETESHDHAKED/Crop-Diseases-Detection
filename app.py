import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model('model.h5')

classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, f.filename)
        f.save(file_path)

        img = image.load_img(file_path, target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        preds = model.predict(x)
        result = classes[np.argmax(preds)]
        
    return render_template('index.html', result=result)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return index()

if __name__ == '__main__':
    app.run(debug=True)
