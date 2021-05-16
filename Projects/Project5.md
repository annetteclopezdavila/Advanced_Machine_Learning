---
layout: page
title: Multi-Label Classification of Fashion Products 
subtitle: Capstone Project
cover-img: /assets/img/NNN.PNG
---

This capstone project will attempt to complete Phase 1 of a wardrobe recommendation algorithm. In Phase 1, we will classify images of fashion items by gender, item category, type of item, season worn, colors, year, and usage. In order to classify the image, we will need to create a multi-label system rather than a binary label system. Then, a CNN will be used to predict classifications.

# Introducing the Data
In order to make a prototype model, I have found a Fashion Product Image Dataset. This dataset can be downloaded from [Kaggle](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset). Due to time, memory, and RAM constrictions, we will be using the smaller version of the dataset (280 MB instead of 15GB). 

~~~
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import pandas as pd 
import os 

PATH = "/kaggle/input/fashion-product-images-dataset/fashion-dataset/fashion-dataset/"
print(os.listdir(PATH))

#create df
df = pd.read_csv(PATH + "styles.csv",nrows=20000, error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)
df.head(10)
~~~
![image](https://user-images.githubusercontent.com/67920563/116352805-bd38a700-a7c3-11eb-9a23-f946925bc803.png)



This dataset is a collection of 44,000 products which are classified into 7 categories: Gender, a masterCategory, articleType, baseColour, season, year, and usage. Within these categories, the images are classified into specific labels. Below are the unique labels within each category in a sample size of 20,000 images:
~~~
target = ['gender', 'masterCategory', 'subCategory', 'articleType',
       'baseColour', 'season', 'year', 'usage']
for col in target:
    print(col)
    print(df[col].unique())
    print('-------------------------')
~~~
![image](https://user-images.githubusercontent.com/67920563/116353098-32a47780-a7c4-11eb-913b-250eefa70c15.png)

For this project, the input variable will be images while the output variable (target) will be the above categories. Because we have more than one target classification, we will need to create a system in which an image may be classified as multiple labels. Because the labels are not mutually exclusive, we cannot formulate a multi-class problem. 

Below is an example of what a multi label/multi class neural network may look like:
![image](https://user-images.githubusercontent.com/67920563/116484552-4cdb6580-a857-11eb-83ab-fa3b2c0e0a32.png)


## Computer Vision & FAST Methods

In this project, we will be using computer vision to categorize and detect products. The following images are example images from the dataset:
~~~
from matplotlib import pyplot as plt
import cv2

for i in range(1, 10):
    
    thisId = str(df[i:i+1].id.values[0])
    
    imageName = '/kaggle/input/fashion-product-images-small/myntradataset/images/'+ thisId +'.jpg'
    image = cv2.imread(imageName)
    image = RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.title(f'Image {thisId}')
    plt.show()
~~~
![image](https://user-images.githubusercontent.com/67920563/116353296-82833e80-a7c4-11eb-8c15-e9ca7c2e0187.png)
![image](https://user-images.githubusercontent.com/67920563/116353321-8dd66a00-a7c4-11eb-8cf2-61af3179b311.png)
![image](https://user-images.githubusercontent.com/67920563/116353345-962ea500-a7c4-11eb-9627-d1dab75c264b.png)

One  method in which a machine can detect products is by applying a corner detection test known as features from accelerated segment test (FAST). This method extracts corner points by using a circle of sixteen pixels around the point in question and uses a brightness threshold value to identify whether pixels around the point are white space. 

We can apply a FAST algorithm to the example images to visualize this method:
~~~
from matplotlib import pyplot as plt
import cv2

for i in range(1, 10):
    
    thisId = str(df[i:i+1].id.values[0])
    
    imageName = '/kaggle/input/fashion-product-images-small/myntradataset/images/'+ thisId +'.jpg'
    image = cv2.imread(imageName)
    image = RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    fast = cv2.FastFeatureDetector_create(50)
    kp = fast.detect(image,None)
    img2 = cv2.drawKeypoints(image, kp, None, color=(255,0,0))
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    fast_image=cv2.drawKeypoints(image,kp,image)
    plt.imshow(fast_image);plt.title('FAST Detector')
    plt.show()
~~~
![image](https://user-images.githubusercontent.com/67920563/116354217-173a6c00-a7c6-11eb-840b-2a535a6618bc.png)
![image](https://user-images.githubusercontent.com/67920563/116354242-21f50100-a7c6-11eb-93d9-3bf532c62f24.png)
![image](https://user-images.githubusercontent.com/67920563/116354283-2e795980-a7c6-11eb-80a0-6505f9c7f3d7.png)

# Data Visualizations
Let us explore frequency of labels in each category:
### Master Category
In the following graph, we can see that the majority of the images contain apparel (~9,000) with accessories coming in second (~5,000). Apparel labels are slightly less than half of the sampled group.
~~~
plt.figure(figsize=(7,20))
df.masterCategory.value_counts().sort_values().plot(kind='barh')
~~~
![image](https://user-images.githubusercontent.com/67920563/116354795-edce1000-a7c6-11eb-8408-98dd0c06a4f4.png)

### Article type
In the chart below, our most frequent labels are as follows: tshirt, shirt, casual shoes, watches, kurtas, tops, and handbags. There are almost 3,000 labels for tshirts. All other article types previously listed are between the ranges of 1500 and 1000.
~~~
plt.figure(figsize=(7,20))
df.articleType.value_counts().sort_values().plot(kind='barh')
~~~
![image](https://user-images.githubusercontent.com/67920563/116354681-c37c5280-a7c6-11eb-9e8d-fcb2a0b5b899.png)
### SubCategory
The chart shows that topwear and shoes are the most frequent labels, both having a frequency between 4,000 and 6,000.
~~~
plt.figure(figsize=(7,20))
df.subCategory.value_counts().sort_values().plot(kind='barh')
~~~
![image](https://user-images.githubusercontent.com/67920563/116354844-fd4d5900-a7c6-11eb-9f4b-43c6267530c9.png)

### Season
In the graph below, we see that a little less than half of the items are summer items (~10,000). The second most frequent season label is fall, with about 5,000 images.
~~~
plt.figure(figsize=(7,20))
df.season.value_counts().sort_values().plot(kind='barh')
~~~
![image](https://user-images.githubusercontent.com/67920563/116354984-3259ab80-a7c7-11eb-8629-b226bd59650f.png)

### Year
Below we can see that the most frequent year labels are 2012, 2011, 2016, and 2017 all in respective order of most to least frequent. Items labeled 2012 carry number to about 7,000 with 2011 taking a close second at ~6,000
~~~
plt.figure(figsize=(7,20))
df.year.value_counts().sort_values().plot(kind='barh')
~~~
![image](https://user-images.githubusercontent.com/67920563/116355071-54532e00-a7c7-11eb-9a53-83bd0fa30c08.png)

### Usage
The bar chart is heavily skewed, showing that the majority of images are for casual usage. In fact, out of the 20,000 images, about 15,000 are casual fashionwear.
~~~
plt.figure(figsize=(7,20))
df.usage.value_counts().sort_values().plot(kind='barh')
~~~
![image](https://user-images.githubusercontent.com/67920563/116355129-67fe9480-a7c7-11eb-8e68-3b0bf069e3cc.png)
### Gender
This graph shows that the most frequent gender label is man, with woman coming in close second. Both have about 8,000-9,000 images
~~~
plt.figure(figsize=(7,20))
df.gender.value_counts().sort_values().plot(kind='barh')
~~~
![image](https://user-images.githubusercontent.com/67920563/116355208-88c6ea00-a7c7-11eb-8fb3-1bb66095c703.png)

### Color
This graph shows that the most frequent color labels are black, white, blue, brown, gray, and red. Black dominates all other colors, with a frequency of about 4,000.
~~~
plt.figure(figsize=(7,20))
df.baseColour.value_counts().sort_values().plot(kind='barh')
~~~
![image](https://user-images.githubusercontent.com/67920563/116355324-af852080-a7c7-11eb-8f05-595a31bf8dbd.png)

From the charts, we can see that some categories are quite skewed in frequency, thus leaving our model to be susceptible to bias. For example, the color black dominated the color category. If we introduce a fluorescent green item (the least frequent color), the probability that the model will accurately label it is much lower than if we introduced a black item. This is because the model has not been able to train itself equally in all color subcategories. The color category, article type category, and subcategory will be most affected by this uneven data. Other categories such as usage and year may see similar trends due to uneven data. 

# Data Preprocessing
Before creating a model, the data will need to be preprocessed. In previous sections, we have downloaded the dataset, created a dataframe, and established target categories. We must now process the image dataset. We must be able to read all images, reset them to a particular size, and insert them into a data list.
~~~
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

#Create Data List
data = []

#Resizing Numbers
IX = 80
IY = 60

#List holding invalid images
invalid_ids = []

#Read and Resize Images

for name in df.id:
    try:
        image = cv2.imread('/kaggle/input/fashion-product-images-small/myntradataset/images/'+str(name)+'.jpg')
        image = cv2.resize(image, (IX,IY) )
        image = img_to_array(image)
        data.append(image)        
    except: 
        # Images for certain ids are missing, so they are not added to the dataset  
        invalid_ids.append(name)
~~~
Now that we have a list of resized images, we must move on to processing the target labels. We must go through each row and add each labels to a list while making sure each label combination is kept together.
~~~
labels = []

for index, row in df.iterrows():
     #invalid ids
    if row['id'] in invalid_ids:
        continue
     
    tags = []
    
    # go through each column in the specified row and add to list
    for col in target:
        tags.append(row[col])

#append the sublist to the labels list
    labels.append(tags)
~~~
We can convert both the data and labels into numpy arrays:
~~~
import numpy as np

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print(labels)
~~~
Our Labels Array will now look as follows:

![image](https://user-images.githubusercontent.com/67920563/116483836-e275f580-a855-11eb-8aa6-4550d6aa4d14.png)

When working with categorical data, we often need to one hot encode our data into numbers. One hot encoding will assign binary vectors to categorical data. On the first attempt, the entire dataset was one-hot encoded, but the method expanded the dataframe to over 222 columns. Because we are one hot encoding the target variables, we can use the Label Binarizer from Sklearn. In this project we have multiple categories, and thus must use the MultiLabelBinarizer to one hot encode our categories.
~~~
#binary vectors for each row
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

print(mlb.classes_)
print(labels[0])
~~~

![image](https://user-images.githubusercontent.com/67920563/116485241-b3ad4e80-a858-11eb-8209-690580d0dd1d.png)

# Model
For this classification problem, we will be using a convolutional neural network. We must first split the training and test data. For this model, we will be using an 80:20 split. A typical CNN design begins with feature extraction and finishes with classification. Feature extraction is performed by alternating convolution layers with subsambling layers. Classification is performed with dense layers followed by a final softmax layer.

~~~
from sklearn.model_selection import train_test_split

# splitting data into testing and training set 
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.2, random_state=2021)
~~~
Let us load the necessary libraries:
~~~
from numpy import mean
from numpy import std
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.optimizers import SGD, Adam
~~~
We must also define the input shape of the pictures in the Neural Network. These will signify the shape and colors used in the pictures. 2 signifies black and white, 3 signifies the color wheel.
~~~
inputShape = (IY, IX, 3) #shape,shape,color
~~~
## Model 1
Before we attempt a model, let us consider possible factors. Because this is a classification problem, we will have one output neuron per class. We must also choose activation functions for hidden and output layers and must use the binary cross-entropy loss function. We can 
~~~
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding="same", input_shape=inputShape),
  tf.keras.layers.MaxPooling2D(2, 2), #downsize
  tf.keras.layers.Dropout(0.25),   #to prevent having dead neurons
  
  tf.keras.layers.Dense(64,  activation='relu'),    
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='sigmoid'),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer = optimizer,
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
              
history= model.fit(trainX, trainY, epochs=100, steps_per_epoch = 1, batch_size = 5)
model.evaluate(testX, testY)
~~~
![image](https://user-images.githubusercontent.com/67920563/118382463-e75fd680-b5c3-11eb-905d-c27fe91ff35b.png)
![image](https://user-images.githubusercontent.com/67920563/118382466-f6df1f80-b5c3-11eb-8f5d-93213753ac16.png)


## Model 2
~~~
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding="same", input_shape=inputShape),
  tf.keras.layers.MaxPooling2D(2, 2), #downsize
  tf.keras.layers.Dropout(0.25),   #to prevent having dead neurons
  
  tf.keras.layers.Dense(64,  activation='relu'),    
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation=tf.nn.softmax),
  tf.keras.layers.Dense(1)
])

from keras.preprocessing.image import ImageDataGenerator
data_augmentation = True
batch_size=1
epochs=1
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(optimizer = optimizer,
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

#history= model.fit(trainX, trainY, epochs=100, steps_per_epoch = 1, batch_size = 5)
x_train = trainX.astype('float32')
x_test = testX.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, trainY,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, testY),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, trainY,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, testY),
                        workers=8)
# Score trained model.
scores = model.evaluate(x_test, testY, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
~~~
![image](https://user-images.githubusercontent.com/67920563/118379321-86c2a080-b5a7-11eb-807a-fbcc74728b86.png)
















