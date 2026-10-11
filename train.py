import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

zip_name = 'archive.zip'

with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall('.')
print("Extract ho gayi")

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train = train_datagen.flow_from_directory('PlantVillage', target_size=(64,64), batch_size=16, class_mode='categorical', subset='training')
val = train_datagen.flow_from_directory('PlantVillage', target_size=(64,64), batch_size=16, class_mode='categorical', subset='validation')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=3)
model.save('model.h5')
print("MODEL BAN GAYA!")
