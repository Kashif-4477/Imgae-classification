
# Natural Scenes Classification: ResNet50, EfficientNetV2, MobileNetV2, VGG16

This project involves classifying natural scenes using deep learning models ResNet50, EfficientNetV2, MobileNetV2, and VGG16. Researchers, machine learning engineers, and enthusiasts interested in image classification can benefit from comparing the performance of these models. Each architecture, such as ResNet50 for depth, EfficientNetV2 for efficiency, MobileNetV2 for lightweight deployment, and VGG16 for simplicity, offers unique characteristics. The project aims to provide insights for selecting the most suitable model for natural scenes classification based on specific requirements.


## Google colab and drive

To link your drive to google colab, download the datasat, create a directory for the datasat zip file and unzipping the file inside the directory run the following set of commands 

```bash
  from google.colab import drive
drive.mount('/content/drive')

from google.colab import files

uploaded = files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
# Download the Intel Image Classification dataset
!kaggle datasets download -d puneet6060/intel-image-classification
!unzip -q intel-image-classification.zip -d ./intel-image-classification


```


## Import important libraries


```bash
  import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetV2B2, MobileNetV2
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.utils import load_img
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import plot_model

```


## Load the datasat

 Some information about the dataset is included here for clarity:

This dataset contains around 25k images of size 150x150 distributed under 6 categories.

{'buildings' -> 0, 'forest' -> 1, 'glacier' -> 2, 'mountain' -> 3, 'sea' -> 4, 'street' -> 5 }

The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.

```bash
  train_path = '/content/intel-image-classification/seg_test/seg_test'
test_path = '/content/intel-image-classification/seg_train/seg_train'

```


## Create ImageDataGenerator

 Define an ImageDataGenerator for augmenting training data with rescaling, zooming,shifting, and flipping. Two generators, train_generator and val_generator, are created to generate batches of training and validation data, respectively, with specified parameters such as target size, batch size, and class mode. The comment lines provide context for each step in the code.

```bash
  # Importing necessary modules
from keras.preprocessing.image import ImageDataGenerator

# Creating an ImageDataGenerator for data augmentation during training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.25,
    validation_split=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

# Creating a training data generator with specified parameters
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=128,
    subset='training',
    shuffle=True,
    class_mode="categorical"
)

# Creating a validation data generator with similar settings as training but for validation subset
val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=128,
    class_mode="categorical",
    shuffle=False,
    subset="validation"
)
```


## Create TestDataGenarator

A separate test_datagen is defined to rescale the test data. Then, a test_generator is created to generate batches of test data with specified parameters such as target size, batch size, and class mode. The shuffle parameter is set to False to maintain the order of data for accurate evaluation. The comments provide a concise explanation of each step.

```bash
# Creating an ImageDataGenerator for rescaling test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Creating a test data generator for evaluating model performance
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=128,
    shuffle=False,
    class_mode="categorical"
)
```


## Labels

A dictionary named labels is created to map numerical class indices to their corresponding class labels using the train_generator information. The subsequent code prints the label mappings for better interpretation of the numerical class indices. The comments provide a clear description of each step

```bash
labels = {value: key for key, value in train_generator.class_indices.items()}

print("Label Mappings for classes \n")
for key, value in labels.items():
    print(f"{key} : {value}")
values = list(labels.values())
```
## Display sample images

Create a subplot grid for visualizing sample images and their corresponding labels
```bash
import matplotlib.pyplot as plt
import numpy as np

# Creating subplots for visualizing sample images
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
idx = 0

# Looping through rows and columns to display images with corresponding labels
for i in range(4):
    for j in range(4):
        # Extracting label for the current image
        label = labels[np.argmax(train_generator[0][1][idx])]
        
        # Setting title with the extracted label
        ax[i, j].set_title(f"{label}")
        
        # Displaying the image
        ax[i, j].imshow(train_generator[0][0][idx][:, :, :])
        
        # Turning off axis labels for better visualization
        ax[i, j].axis("off")
        idx += 1

# Ensuring a tight layout for better aesthetics
plt.tight_layout()

# Displaying the plot
plt.show()
```
![output](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/fb480525-8d20-4c32-bac3-6943c6b08144)

## Define a generic model for later customization
In the following sections, 4 pretrained models were used to classify the images from this dataset. The models are: ResNet50, EfficientNetV2, MobileNetV2, VGG16. The customized top is the same for all 4 models and consists of a GlobalAveragePooling layer, followed by a Dense(128) layer, a Dropout layer (0.2) and a final Dense layer for classification. By using the same customized top, we can compare the performance of the 4 pretrained models later.

It is useful to define a function that takes in as input the choice of the base model, since the architecture for all models is the same. The output of the function will be the trained model, the accuracy of the trained model evaluated on the test set, and the confusion matrix.

```bash
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.ckpt',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
]
def pretrained_custom_model(base_model, eps):
    pretrained_base_model = base_model(include_top = False,
                       weights = 'imagenet',
                       input_shape = (150,150,3))

    for layer in pretrained_base_model.layers:
        layer.trainable = False

    inputs = Input(shape=(150, 150, 3))
    x = pretrained_base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x =Dense(6, activation='softmax')(x)
    custom_model = Model(inputs, x)
    custom_model.summary()

    #plot_model(custom_model, show_shapes = True)

    custom_model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

    history = custom_model.fit(train_generator, validation_data = val_generator,
                                      epochs=eps,callbacks=callbacks)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    #Plot loss and accuracy curves
    fig, ax = plt.subplots(2, 1, figsize = (8, 8))
    epochs = range(1, len(acc) + 1)
    ax[0].plot(epochs, acc, 'b--', label='Training acc')
    ax[0].plot(epochs, val_acc, 'r', label='Validation acc')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()

    ax[1].plot(epochs, loss, 'b--', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()

    #Prediction
    pred = custom_model.predict(test_generator)
    y_test = test_generator.classes
    y_pred = np.argmax(pred, axis=1)
    model_acc = accuracy_score(y_test,y_pred)

    #Confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize = (6,6))
    sns.heatmap(cm,
            annot=True, xticklabels=values,
            yticklabels=values,
            cmap='viridis')
    plt.title('Confusion matrix of the trained model')

    print(f'The accuracy of the model is:{model_acc}')
    return custom_model, model_acc, cm
```

## ResNet50: Pretrained model + customized top
Run the RestNet model using 15 epochs:

```bash
resnet50_model = pretrained_custom_model(ResNet50, 15)
```
![__results___19_1](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/52fe1af4-2953-4aa2-aefb-cb1ed758bacf)

![confmt](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/e08554ac-5348-4d6b-a9f6-1efd6f5e1caa)


## EfficientNetV2: Pretrained model + customized top
Use 15 epochs and test the accuracy:

```bash
effnet_model = pretrained_custom_model(EfficientNetV2B2, 15)
```
![eff](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/1c033269-06c4-4bc7-8ea7-8d20d7fb520c)
![coneff](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/cb6d10d6-56b3-45c5-a67f-a46429805eaf)

## MobileNetV2: Pretrained model + customized top
 
 ```bash
mobnet_model = pretrained_custom_model(MobileNetV2, 15)
```
![mob](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/bb3791c8-7192-4f47-a2a1-847d4f3ef1ae)

![mobcnff](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/2b17f0db-cbdc-4c69-9dfc-7cfed4bc2e1b)
##  VGG16 model: Pretrained model + customized top
```bash
vgg16_model = pretrained_custom_model(VGG16, 15)
```
![vgg](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/52263c7b-f409-4ef9-a493-ae0fe11af54a)

![convgg](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/8f44483f-4444-4bbb-a9c5-1feed1b71d03)
## Comparison of accuracy

```bash
d  = pd.DataFrame({'Model': ['EfficientNetV2', 'ResNet50','VGG16', 'MobileNetV2'],
      'Accuracy': [effnet_model[1], resnet50_model[1],  vgg16_model[1],  mobnet_model[1]]})
d = d.set_index('Model')

f, ax = plt.subplots(figsize = (7,5))
d['Accuracy'].sort_values(ascending = True).plot(ax = ax)
ax.axvline("EfficientNetV2", color="green", linestyle="dashed")
ax.axvline("ResNet50", color="green", linestyle="dashed")
ax.axvline("VGG16", color="green", linestyle="dashed")
ax.axvline("MobileNetV2", color="green", linestyle="dashed")
plt.title('Accuracy of various methods in ascending order');
```

![acc](https://github.com/Kashif-4477/Site-Suitability-with-AHP-Analysis/assets/98015675/0839d7e7-5a0d-4a77-b0de-57618b501c74)
