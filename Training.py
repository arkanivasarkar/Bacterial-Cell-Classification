import numpy as np
np.random.seed(1000)
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential



SIZE = 512
dataset = []
label = []
img_dir = 'Desktop/train_images/'

#iterating through all images in gm_pos and gm_neg folder and resizing them to 512 x 512 and labelling gm_pos images as 0 and gm_neg as 1
gmpos_images = os.listdir(img_dir + 'gm_pos/')
for i, image_name in enumerate(gmpos_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(img_dir + 'gm_pos/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)


gmneg_images = os.listdir(img_dir + 'gm_neg/')
for i, image_name in enumerate(gmneg_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(img_dir + 'gm_neg/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)


#defining CNN model
model = None
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(activation = 'relu', units=512))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Dense(activation = 'relu', units=256))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))
model.add(Dense(activation = 'sigmoid', units=2))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

#splitting data into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 0)

#training the model
history = model.fit(X_train, 
                    y_train, 
                    batch_size = 32, 
                    verbose = 1, 
                    epochs = 10,      
                    validation_split = 0.1,
                    shuffle = False)

#model accuracy
print("Validation_Accuracy: {:.2f}%".format(model.evaluate(X_test, y_test)[1]*100))
