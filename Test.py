import numpy as np
np.random.seed(1000)
import os
import cv2
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model

#load trained model
model = load_model('model.h5')

SIZE = 512

#classify single image
img=load_img('Bac cell classification/test_images/gm_pos1.jpg',target_size=(SIZE,SIZE))
x = img_to_array(img)
x = x/255
x = x.reshape((1,) + x.shape)

model.predict(x)

a = np.argmax(model.predict(x), axis=1)
if(a==1):
    print("gram positive")
else:
    print("gram negative")