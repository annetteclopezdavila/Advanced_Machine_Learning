---
layout: page
title: Capstone Project
subtitle: Multi-Label Classification of Fashion Products
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

One way in method in which a machine can detect products is by applying a corner detection test known as features from accelerated segment test (FAST). This method extracts corner points by using a circle of sixteen pixels around the point in question and uses a brightness threshold value to identify whether pixels around the point are white space. 

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






