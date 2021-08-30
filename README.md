# Bacterial-Cell-Classification

This repository contains the implementation of a convolutional neural network for classifying bacterial cells into gram positive and gram negative.

Gram positive bacteria take up Gram stain in their cell wall and hence are deep violet in colour, whereas gram negative ones do not take up Gram stain and hence are pink due to safranin stain.

[![Picture5.png](https://i.postimg.cc/3N1pYCfR/Picture5.png)](https://postimg.cc/wRt3kJp8)



## Methods
Images available for developing the model were very less. Hence, data augmentation was done before training.

The following steps were applied before training the model:
- Image Normalisation
- Data Augmentation

Around 1500 images were generated using data augmentation for each category. 500 were kept for testing.

The training dataset was then split into 80:20 ratio for training and validation.

Adam optimizer with a learning rate of 0.001 was used as optimizer and categorical crossentropy was used as the loss function. The models were trained for 10 epochs with a batch size of 32, using NVIDIA MX-150 GPU. 

## Results
The performance of the model was evaluated using the test dataset.
97.56% classification accuracy was obtained.


Transfer Learning was also implemented using VGG16, but it did not gave significant results.


The trained CNN model is present in `Trained model` folder.



