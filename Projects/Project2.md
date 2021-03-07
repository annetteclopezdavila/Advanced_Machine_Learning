---
layout: page
title: Project 2
---
Comparing the performance of Regularization Methods on different data sets. 

# Introducing the Data
## Costa Rican Household Poverty Level Prediction- IDB
This dataset was scavenged from an old competition on Kaggle asking users to classify household poverty levels on a scale of 1 to 4 based on demographic data. 
The dataset originally had 142 features with 9557 data entries, but was reduced during pre-processing and clean up phases. The dataset's features before preprocessing were as follows:

![image](https://user-images.githubusercontent.com/67920563/110225039-495adc00-7eaf-11eb-8387-d33a6134dd3c.png)


## Preprocessing
9557 data entries were reduced to 156 entries due to the amount of values that were missing from the set. Although NaN values can be either estimated with KNN algorithms or inputed with designated values, an enormous data set was not necessary for the primary purpose of this project. Four columns were also dropped due to them being object type variables. Rather than deleting these columns, we could have assigned a representative number (for example, Yes=0 No=1). The final preprocessed dataset had 156 rows and 138 columns

~~~
#Import Libraries
import numpy as np
import pandas as pd

#Create Dataset
df = pd.read_csv('/content/drive/MyDrive/costa-rican-household-poverty-prediction/train.csv')

#Drop rows with NaN values
df2=df
df2=df2.dropna()

#Exclude type: object
df2 = df2.select_dtypes(exclude=['object'])

#Convert all objects to Numeric
df2=df2.apply(pd.to_numeric)

#Define X variable/features
X=df2.drop('Target', axis=1)

#Define Y variable/target
y=df2['Target']
~~~





