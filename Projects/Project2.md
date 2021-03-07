---
layout: page
title: Project 2
---
Comparing the Performance of Regularization Methods. 

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

We can also see the data distribution in our target:
~~~
y.value_counts()
~~~
![image](https://user-images.githubusercontent.com/67920563/110246781-fd537a00-7f36-11eb-80be-4e45460ac48c.png)

We can see that the dataset hardly has data recorded for level 1 and 2 poverty levels, thus making it more likely that our model will not accurately predict these levels of poverty. It is likely that there were many unknown values in these categories that were deleted in our preprocessing phase. Thus, we should reconsider how we process our NaN values in the future.

## Correlation

In order to later understand regularization techniques, the correlation of the features must be analyzed.

~~~
corr_matrix=np.corrcoef(X)
corr_matrix

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(200,200))
sns.heatmap(X.corr(), vmax=1, vmin=-1, cmap="spring", annot=True,fmt='.2f')
plt.show()
~~~

![image](https://user-images.githubusercontent.com/67920563/110245203-376d4d80-7f30-11eb-95a8-e4e7c4f7b0c9.png)


Although it is somewhat hard to tell about the nature of specific correlations due to the anount of features in the dataset, we can see that the majority of the data has a mild correlation relationship (dominance of peachy coral pink color). 

# Regularization and Feature Selection
## Finding the Weights Mathematically without Regularization

Let us assume we have a model Y with equation Y=F(X1, X2,...,Xp). In weighted linear regression, this equation is multiplied by certain weights which define the sparsity pattern of the data. An error term is also added into the equation:

![image](https://user-images.githubusercontent.com/67920563/110245246-5a97fd00-7f30-11eb-8237-5d0c5e905a61.png)


Our goal is to minimize the squared error by obtaining weights that will do so. Our error term will be found by subtracting the predicted valaues from the observed values and then squared.

![image](https://user-images.githubusercontent.com/67920563/110245255-64b9fb80-7f30-11eb-99be-54c820605cce.png)

The sum of squared errors can also be expressed in matrix form, since our data sets tend to have hundreds/thousands of items. Thus, in matrix notation, the sum of squared errors is the same as transposing two error vectors.

![image](https://user-images.githubusercontent.com/67920563/110245261-6d123680-7f30-11eb-8469-6d8ab378cf77.png)

Putting in the equatoin for error we have: 

![image](https://user-images.githubusercontent.com/67920563/110245268-74d1db00-7f30-11eb-9156-0e7130ffec68.png)

Then, the partial derivatives of the weights are taken:

![image](https://user-images.githubusercontent.com/67920563/110245159-10af1700-7f30-11eb-89b4-516ac163064e.png)

And the equation is set to zero to obtain the global minimum:

![image](https://user-images.githubusercontent.com/67920563/110245366-ba8ea380-7f30-11eb-8613-6a577782912f.png)

Thus, this becomes are solution to the linear regression.

## Linear Regression Application of the IDB Dataset

Note: I was so excited about finding this dataset that I did not realize that it addressed a classification problem rather than a linear regression. It was not until I finished the majority of the coding that I realized that the linear models predicts on a continuous scale rather than a discrete scale. Thus, I was seeing predicted values like 1.6 rather than 2. In order to try to make up for this error, I decided to round the values to their integers. This led to a lot more error than would have occurred with a different X and y. I have decided to keep the error in since this is the second time I have made this error in a Machine Learning Project, so I would like to keep this as a reminder to understand the dataset before running into the application phase.

~~~
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)

lm = LinearRegression()

#fit the variables to model
lm.fit(X_train, y_train)

#predict output by inputting test variable
yhat_lm=lm.predict(X_test)

#Turn continuous variables into discrete
ylist=[]
for output in yhat_lm:
  if output<4.5:
    output=round(output)
    ylist.append(output)
  else:
    output=4
    ylist.append(output)
    
#round
y_hat_lm_rounded=np.array(ylist)

#MAE 
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(lm.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_lm_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
Predicted Values:
![image](https://user-images.githubusercontent.com/67920563/110247412-30e3d380-7f3a-11eb-8d89-5ed59080d54f.png)


![image](https://user-images.githubusercontent.com/67920563/110246942-b9ad4000-7f37-11eb-948e-7d66ffc78e59.png)

~~~
lm.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110247032-13156f00-7f38-11eb-9b03-b5f50334989d.png)

~~~
#max coefficient (range)
max(lm.coef_)

#min coefficient
min(lm.coef_)
~~~
Max:
![image](https://user-images.githubusercontent.com/67920563/110247039-1f013100-7f38-11eb-8ae3-7b088b83bb2b.png)

Min:
![image](https://user-images.githubusercontent.com/67920563/110247040-232d4e80-7f38-11eb-93b0-4f8cea58cd88.png)

We can see from the data that the our coefficients have a rather wide range and our MAE is high. Let us see what happens when we apply regularization to our model.

## Regularization Embedded Models
When we are working with large datasets such as this one, too many features may create *bias* and *variance* in our results. Bias is defined as the inability for the model to capture the true relationship of the data and while variance is defined as the difference in fit between the training and testing sets. For example, a model has high variance if the model predicts the training set very accurately but fails to do so in the testing set (overfitting). In order to reduce bias and variance, feature selection, regularization, boosting, and bagging techniques can be applied.

Feature selection is defined as the selection of features that best contribute to the accuracy of the model. Regularization will add constraints that lower the size of the coefficients in order to make the model less complex and avoid it from fitting variables that are not as important. This will penalize the loss function by adding a regularization term and minimize the sum of square residuals with constraints on the weight vector.

## Lasso Regression/ L1 Regularization
![image](https://user-images.githubusercontent.com/67920563/110245856-03dff280-7f33-11eb-8569-ec6d2f9211c9.png)

Lasso Regression will produce a model with high accuracy and a subset of the original features. Lasso regression puts in a constraint where the sum of absolute values of coefficients is less than a fixed value. Thus, it will lower the size of the coefficients and lead some features to have a coefficient of 0, essentially dropping it from the model.

Looking back at our model without regularization, we saw that our coefficients were found with the formula:
![image](https://user-images.githubusercontent.com/67920563/110245366-ba8ea380-7f30-11eb-8613-6a577782912f.png)

L1 regularization's loss function is defined by:
![image](https://user-images.githubusercontent.com/67920563/110245501-6637f380-7f31-11eb-9823-e8ccd62af23f.png)

otherwise known as:

![image](https://user-images.githubusercontent.com/67920563/110245529-8e275700-7f31-11eb-84e8-581085748f7a.png)

By adding in an extra cost term, the weights will be penalized. Lasso Regression will find the closed form solution to this equation in order to derive the weights. 

This type of regularization should not be applied to a dataset with a low number of features as it will possibly drop features that are essential to the model. Lasso regularization also does not work well with features that are highly correlated, as it may drop correlated groups. The solution will be sparse as a large proportion of features will have a weight of zero.

## Lasso Regression Application
~~~
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.01)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
ls.fit(X_train,y_train)
yhat_ls=ls.predict(X_test)

#Turn continuous variables into discrete
ylist=[]
for output in yhat_ls:
  if output<4.5:
    output=round(output)
    ylist.append(output)
  else:
    output=4
    ylist.append(output)

y_hat_ls_rounded=np.array(ylist)
~~~
![image](https://user-images.githubusercontent.com/67920563/110247383-042fbc00-7f3a-11eb-9b28-ed093ed0c37f.png)

~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(ls.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_ls_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
![image](https://user-images.githubusercontent.com/67920563/110247468-64266280-7f3a-11eb-849b-23d45de3ba92.png)
~~~
print('Coefficients')
ls.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110247447-5c66be00-7f3a-11eb-886c-4ad787557cbf.png)

~~~
#get features that work
list=[]
count=-1
counted_list=[]
A=ls.coef_
for a in A:
  count=count+1
  if a!=0:
    list.append(a)
    counted_list.append(count)

#Features
for c in counted_list:
  print(X.columns[c])
~~~
![image](https://user-images.githubusercontent.com/67920563/110247485-7ef8d700-7f3a-11eb-9782-3339ffb6c32e.png)

~~~
#test other alphas
import matplotlib.pyplot as plt
maeLS=[]
for i in range(100):
  ls = Lasso(alpha=i)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
  ls.fit(X_train,y_train)
  yhat_ls=ls.predict(X_test)
  #Turn continuous variables into discrete
  ylist=[]
  for output in yhat_ls:
    if output<4.5:
      output=round(output)
      ylist.append(output)
    else:
      output=4
      ylist.append(output)

  y_hat_ls_rounded=np.array(ylist)

  maeLS.append(mean_absolute_error(y_test, y_hat_ls_rounded))
  plt.scatter(range(100),maeLS)
  ~~~
  
  
![image](https://user-images.githubusercontent.com/67920563/110247573-eb73d600-7f3a-11eb-90ca-3d6ebe6565d9.png)

## Standardized Lasso Regression Application

~~~
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ls = Lasso(alpha=0.01)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
ls.fit(ss.fit_transform(X_train),y_train)
yhat_ls=ls.predict(ss.fit_transform(X_test))
#Turn continuous variables into discrete
ylist=[]
for output in yhat_ls:
  if output<4.5:
    output=round(output)
    ylist.append(output)
  else:
    output=4
    ylist.append(output)

y_hat_ls_rounded=np.array(ylist)
y_hat_ls_rounded
~~~
![image](https://user-images.githubusercontent.com/67920563/110247700-8a98cd80-7f3b-11eb-9928-2f5165e914bc.png)

~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(ls.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_ls_rounded)
print("MAE = {:,.2f}".format(mae))
~~~~
![image](https://user-images.githubusercontent.com/67920563/110247734-b9af3f00-7f3b-11eb-8143-df049088af10.png)

~~~
print('Coefficients')
ls.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110247769-da779480-7f3b-11eb-8b2d-346b8de26855.png)

~~~
#get features that work
list=[]
count=-1
counted_list=[]
A=ls.coef_
for a in A:
  count=count+1
  if a!=0:
    list.append(a)
    counted_list.append(count)

#Features
for c in counted_list:
  print(X.columns[c])
~~~
![image](https://user-images.githubusercontent.com/67920563/110247803-0bf06000-7f3c-11eb-899d-bb979a780db6.png)

~~~
#test other alphas
import matplotlib.pyplot as plt
maeSLS=[]
for i in range(100):
  ss = StandardScaler()
  ls = Lasso(alpha=i)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
  ls.fit(ss.fit_transform(X_train),y_train)
  yhat_ls=ls.predict(ss.fit_transform(X_test))
  #Turn continuous variables into discrete
  ylist=[]
  for output in yhat_ls:
    if output<4.5:
      output=round(output)
      ylist.append(output)
    else:
      output=4
      ylist.append(output)

  y_hat_ls_rounded=np.array(ylist)

  maeSLS.append(mean_absolute_error(y_test, y_hat_ls_rounded))
  plt.scatter(range(100),maeSLS)
  ~~~
![image](https://user-images.githubusercontent.com/67920563/110248069-4c041280-7f3d-11eb-8143-ddae046d5d5c.png)


## Ridge Regression Application

~~~
#Apply Ridge
from sklearn.linear_model import Ridge

lr = Ridge(alpha=0.01)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
lr.fit(X_train,y_train)
yhat_lr=lr.predict(X_test)
#Turn continuous variables into discrete
ylist=[]
for output in yhat_lr:
  if output<4.5:
    output=round(output)
    ylist.append(output)
  else:
    output=4
    ylist.append(output)

y_hat_lr_rounded=np.array(ylist)
~~~
![image](https://user-images.githubusercontent.com/67920563/110248492-7ce54700-7f3f-11eb-9569-90d19662d405.png)

~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(lr.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_lr_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
![image](https://user-images.githubusercontent.com/67920563/110248580-e1a0a180-7f3f-11eb-94fd-c1133636c995.png)

~~~
print('Coefficients:')
lr.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110248619-0c8af580-7f40-11eb-9439-daea4be4e5a1.png)

~~~
max(lr.coef_)
min(lr.coef_)
~~~
Max:
![image](https://user-images.githubusercontent.com/67920563/110248636-2593a680-7f40-11eb-89a9-5bde7fa59e93.png)

Min:
![image](https://user-images.githubusercontent.com/67920563/110248650-33492c00-7f40-11eb-8429-76a0556296e0.png)

~~~
import matplotlib.pyplot as plt
maeLR=[]
for i in range(1000):
  lr = Ridge(alpha=i)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
  lr.fit(X_train,y_train)
  yhat_lr=lr.predict(X_test)
  #Turn continuous variables into discrete
  ylist=[]
  for output in yhat_lr:
    if output<4.5:
      output=round(output)
      ylist.append(output)
    else:
      output=4
      ylist.append(output)

  y_hat_lr_rounded=np.array(ylist)
  
  maeLR.append(mean_absolute_error(y_test, y_hat_lr_rounded))
  plt.scatter(range(1000),maeLR)
  ~~~
  
  ![image](https://user-images.githubusercontent.com/67920563/110248715-8de28800-7f40-11eb-9040-dcb123e49954.png)

## Standardized Ridge Regression Application
~~~
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

lr = Ridge(alpha=0.01)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
lr.fit(ss.fit_transform(X_train),y_train)
scaled_yhat_lr=lr.predict(ss.fit_transform(X_test))
#Turn continuous variables into discrete
ylist=[]
for output in scaled_yhat_lr:
  if output<4.5:
    output=round(output)
    ylist.append(output)
  else:
    output=4
    ylist.append(output)

y_hat_lr_rounded=np.array(ylist)
~~~
![image](https://user-images.githubusercontent.com/67920563/110248784-d1d58d00-7f40-11eb-8a0e-cb33c676b82e.png)

~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(lr.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_lr_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
![image](https://user-images.githubusercontent.com/67920563/110248808-eade3e00-7f40-11eb-8580-2cbb987d49a8.png)
~~~
print('Coefficients:')
lr.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110248842-03e6ef00-7f41-11eb-953d-31548e7a768d.png)
~~~
max(lr.coef_)
min(lr.coef_)
~~~
Max:
![image](https://user-images.githubusercontent.com/67920563/110248909-60e2a500-7f41-11eb-909a-6090b68cba68.png)

Min:
![image](https://user-images.githubusercontent.com/67920563/110248928-71931b00-7f41-11eb-9f9e-99e8df4bc729.png)

~~~
import matplotlib.pyplot as plt
maeSLR=[]
for i in range(200):
  lr = Ridge(alpha=i)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
  lr.fit(ss.fit_transform(X_train),y_train)
  scaled_yhat_lr=lr.predict(ss.fit_transform(X_test))
  #Turn continuous variables into discrete
  ylist=[]
  for output in scaled_yhat_lr:
    if output<4.5:
      output=round(output)
      ylist.append(output)
    else:
      output=4
      ylist.append(output)

  y_hat_lr_rounded=np.array(ylist)
  
  maeSLR.append( mean_absolute_error(y_test, y_hat_lr_rounded))
  plt.scatter(range(200),maeSLR)
  ~~~
![image](https://user-images.githubusercontent.com/67920563/110248980-bb7c0100-7f41-11eb-820c-d7d435d7863e.png)











