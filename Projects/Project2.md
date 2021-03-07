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

## Correlation

In order to later understand regularization techniques, the correlation of the features must be analyzed.

~~~
corr_matrix=np.corrcoef(X)
corr_matrix

import seaborn as sns
plt.figure(figsize=(200,200))
sns.heatmap(X.corr(), vmax=1, vmin=-1, cmap="spring", annot=True,fmt='.2f')
plt.show()
~~~

#INSERT CORRELATION MATRIX


Although it is somewhat hard to tell about the nature of specific correlations due to the anount of features in the dataset, we can see that the majority of the data has a mild correlation relationship (dominance of peachy coral pink color). 

# Regularization and Feature Selection
## Regularization Embedded Models
When we are working with large datasets such as this one, too many features may create *bias* and *variance* in our results. Bias is defined as the inability for the model to capture the true relationship of the data and while variance is defined as the difference in fit between the training and testing sets. For example, a model has high variance if the model predicts the training set very accurately but fails to do so in the testing set (overfitting). In order to reduce bias and variance, feature selection, regularization, boosting, and bagging techniques can be applied.

Feature selection is defined as the selection of features that best contribute to the accuracy of the model. Regularization will add constraints that lower the size of the coefficients in order to make the model less complex and avoid it from fitting variables that are not as important. This will penalize the loss function by adding a regularization term and minimize the sum of square residuals with constraints on the weight vector.

Let us assume we have a model Y with equation Y=F(X1, X2,...,Xp). In weighted linear regression, this equation is multiplied by certain weights which define the sparsity pattern of the data. An error term is also added into the equation:


#INSERT WEIGHTED LINEAR REGRESSION PIC

Our goal is to minimize the squared error by obtaining weights that will do so. Our error term will be found by subtracting the predicted valaues from the observed values and then squared.

#PUT IN ERROR TERM PICTURE

The sum of squared errors can also be expressed in matrix form, since our data sets tend to have hundreds/thousands of items. Thus, in matrix notation, the sum of squared errors is the same as transposing two error vectors.

#PUT IN TRANSPOSED

Putting in the equatoin for error we have: 

#PUT IN EQUATION TRANSPOSE

Then, the partial derivatives of the weights are taken:
![image](https://user-images.githubusercontent.com/67920563/110245159-10af1700-7f30-11eb-89b4-516ac163064e.png)



## Lasso Regression/ L1 Regularization
#INSERT LASSO EQUATION

Lasso Regression will produce a model with high accuracy and a subset of the original features. Lasso regression puts in a constraint where the sum of absolute values of coefficients is less than a fixed value. Thus, it will lower the size of the coefficients and lead some features to have a coefficient of 0, essentially dropping it from the model.





