from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = load_model('pneu.h5')

def pneu_pred(fp: str):
    img = image.load_img(fp, target_size=(224, 224))
    imagee = image.img_to_array(img)
    imagee = np.expand_dims(imagee, axis=0)
    img_data = preprocess_input(imagee)
    pred = model.predict(img_data)
    if pred[0][0] > pred[0][1]:
        return False
    return True
