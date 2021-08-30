import os
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

#data augmentation parameter
datagen = ImageDataGenerator(
        rotation_range=120,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect')


img_dir = 'E:/Bac cell classification/Bac img/' #image_directory

#augment gram pos bac imgs
gmpos_images = os.listdir(img_dir + 'gram_pos/')
for i, image_name in enumerate(gmpos_images):
    if (image_name.split('.')[1] == 'jpg') or (image_name.split('.')[1] == 'JPG') or (image_name.split('.')[1] == 'jpeg') or (image_name.split('.')[1] == 'tif'):
        img = load_img(img_dir + 'gram_pos/' + image_name)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='E:/Bac cell classification/Bac img/augmented/gm_pos', save_prefix='gmpos', save_format='jpg'):
            i += 1
            if i > 30:
                break


#augment gram neg bac imgs
gmneg_images = os.listdir(img_dir + 'gram_neg/')
for i, image_name in enumerate(gmneg_images):
    if (image_name.split('.')[1] == 'jpg') or (image_name.split('.')[1] == 'JPG') or (image_name.split('.')[1] == 'jpeg') or (image_name.split('.')[1] == 'tif'):
        img = load_img(img_dir + 'gram_neg/' + image_name)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='E:/Bac cell classification/Bac img/augmented/gm_neg', save_prefix='gmneg', save_format='jpg'):
            i += 1
            if i > 30:
                break