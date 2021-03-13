---
layout: page
title: Project 2
subtitle: Comparing the Performance of Regularization Methods
---


# Costa Rican Household Poverty Level Prediction- IDB
## Introducing the Data
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

In order to further verify that the data is weakly correlated, we can apply a one sample t-test with a two sided pvalue. 
~~~
from scipy import stats
def tstat_corr(r,n):
  t = r*np.sqrt(n-2)/np.sqrt(1-r**2)
  pval = stats.t.sf(np.abs(t), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
  print('t-statistic = %6.3f pvalue = %6.4f' % (t, pval))
  
 tstat_corr(0.02,len(X)) #put in an r value from matrix
 ~~~
 One of the lowest correlation values in the heatmap was chosen as the r value. We are left with the following results:
 ![image](https://user-images.githubusercontent.com/67920563/110415773-7f8b8d80-8060-11eb-9b10-009ebf673340.png)
 
 The t-value measures the size of the difference relatie to the variation in the data. The greater t is, the greater the chance is that there is a significant difference between the data. Our p value is extremly high at about 0.8, thus telling us that the data is most likely weakly correlated.

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

Note: I was so excited about finding this dataset that I did not realize that it addressed a classification problem rather than a linear regression. It was not until I finished the majority of the coding that I realized that the linear models predicts on a continuous scale rather than a discrete scale. Thus, I was seeing predicted values like 1.6 rather than 2. In order to try to make up for this error, I decided to round the values to their integers. This led to a lot more error/less accuracy than would have occurred with a different X and y. I have decided to keep the error in since this is the second time I have made this error in a Machine Learning Project, so I would like to keep this as a reminder to understand the dataset before running into the application phase. An SVM would better work for classification problems.

Before applying any regularization, let us examine a linear regression model on the data:

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

# Lasso Regression/ L1 Regularization
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
We can first fit our data to a lasso regression:

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
Our predicted values are:

![image](https://user-images.githubusercontent.com/67920563/110247383-042fbc00-7f3a-11eb-9b28-ed093ed0c37f.png)

Let us find the mean absolute error to compare the model on the training and testing set:
~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(ls.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_ls_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
![image](https://user-images.githubusercontent.com/67920563/110247468-64266280-7f3a-11eb-849b-23d45de3ba92.png)

The weighted coefficients are:
~~~
print('Coefficients')
ls.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110247447-5c66be00-7f3a-11eb-886c-4ad787557cbf.png)

Let us examine the range of the coefficients:

~~~
min(ls.coef_)
~~~
![image](https://user-images.githubusercontent.com/67920563/110425678-be2a4380-8072-11eb-8af3-7378eefe5884.png)

~~~
max(ls.coef_)
~~~
![image](https://user-images.githubusercontent.com/67920563/110425689-c2566100-8072-11eb-981c-3fbfcca260f1.png)

We can see that the range of the coefficients has decreased greatly compared to the non regularized model.

Since Lasso Regression eliminates certain features, we can derive the ones that are left:
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

It can be noticed that we chose an arbitrary penalty term (alpha value) in this example. In order to choose alpha, we must use cross validation. The alpha value that gives us the least variance is the optimal value. We can thus test other alpha values that may make our predictions better:
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
Looking at our data, we can observe that the values are not nearly in the same ranges. Thus, we can standardize our data to attempt to make our model better. We apply the same steps while standardizing our features:

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

We can see that standardizing our data improves our MAE greatly at the same alpha (from 0.32 to 0.23).
~~~
print('Coefficients')
ls.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110247769-da779480-7f3b-11eb-8b2d-346b8de26855.png)

Let us look at the range of our coefficients:
~~~
min(ls.coef_)
~~~
![image](https://user-images.githubusercontent.com/67920563/110426139-8e2f7000-8073-11eb-80c7-4d670fe48835.png)
~~~
max(ls.coef_)
~~~
![image](https://user-images.githubusercontent.com/67920563/110426146-925b8d80-8073-11eb-81cd-743a291b6763.png)

Thus, we can see that the standardized Lasso Regression decreased the range of weights in comparison to the non standardized Lasso Regression.

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

We see a similar trend in the standardized and non-standardized forms of Lasso regression; the MAE plateaus below 0.22 as alpha goes to infinity.

# Ridge Regression/L2 Regularization

![image](https://user-images.githubusercontent.com/67920563/110421480-997e9d80-806b-11eb-97d9-8052466197bb.png)

Ridge Regression shares many conceptual similarities with Lasso Regression; it also adds on a penalty to the loss function. The regularization term is the sum of squares of all the feature weights. Unlike Lasso Regression, this type of regression will make the weights smaller but never zero. Ridge regession is not good for data with a lot of outliers, as it blows up the error differences of the outliers and the regularization term tries to fix it by penalizing the weights. Ridge regression is also better when all the features influence the output and all the weights are roughly the same size. This regularization technique does not offer feature selection and has a non sparse solution. It should be noted that ridge regression can hel solve models in which there are less data points than parameters. Ridge regression will penalize large weight coefficients more than the smaller ones as opposed to Lasso regression which penalizes each coefficient uniformly.

Ridge regression's loss function is defined as:

![image](https://user-images.githubusercontent.com/67920563/110421214-0ba2b280-806b-11eb-8244-81be15b5293d.png)

One can solve for the weighted term of ridge regression by finding the closed form solution using the derivative. We end up with the form:

![image](https://user-images.githubusercontent.com/67920563/110421550-b7e49900-806b-11eb-935f-4a572c281d6c.png)


## Ridge Regression Application
Ridge Regression has a similar application code wise as Lasso Regression:
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

We are able to calculate the MAE on unstandardized data:
~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(lr.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_lr_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
![image](https://user-images.githubusercontent.com/67920563/110248580-e1a0a180-7f3f-11eb-94fd-c1133636c995.png)

We see that unstandardized Ridge and Lasso regularization produce the same MAE for this dataset.
~~~
print('Coefficients:')
lr.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110248619-0c8af580-7f40-11eb-9439-daea4be4e5a1.png)

Our range of weights decreases from the non-regularized linear regression coefficients, but does not decrease as much as Lasso Regression:
~~~
max(lr.coef_)
min(lr.coef_)
~~~
Max:

![image](https://user-images.githubusercontent.com/67920563/110248636-2593a680-7f40-11eb-89a9-5bde7fa59e93.png)

Min:

![image](https://user-images.githubusercontent.com/67920563/110248650-33492c00-7f40-11eb-8429-76a0556296e0.png)

We once again test for optimal alpha values:
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
  
  It can be noted that as alpha gets bigger, y is less sensitive to the features. The optimal alpha value drops below an MAE of 0.20 when alpha is about 250. This so far has presented the best model.

## Standardized Ridge Regression Application
We repeat the process with standardization:
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

The MAE has increased above all models including the non-regularized model, thus showing that this may not be an adequate structure for our data.

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

Although our MAE has increased, the range of values of weight coefficients is still lower than the original model.
~~~
max(lr.coef_)
min(lr.coef_)
~~~
Max:
![image](https://user-images.githubusercontent.com/67920563/110248909-60e2a500-7f41-11eb-909a-6090b68cba68.png)

Min:
![image](https://user-images.githubusercontent.com/67920563/110248928-71931b00-7f41-11eb-9f9e-99e8df4bc729.png)

Trying different penalty values we get:

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

Note that at an alpha value of about 90-150, the MAE decreases to 0.15. This so far has been the lowest recorded MAE, proving that this may indeed be a good model for our data.

# Ridge vs Lasso
Let us more clearly compare the two regularization techniques:

![image](https://user-images.githubusercontent.com/67920563/110428563-75c15480-8077-11eb-8391-70dce71dabc9.png)


# ElasticNet
ElasticNet regression combines L1 and L2 regularization:

![image](https://user-images.githubusercontent.com/67920563/110429057-60005f00-8078-11eb-86e2-68f4411a0672.png)

This particular model combines feature elimination with coefficient reduction to create a better model; ElasticNet will also do well with highly correlated data. When variables are correlated, Ridge regression will shrink the two coefficients towards each other while Lasso will overlook one variable and drop it. ElasticNet will avoid such problems by combining both models.
~~~
from sklearn.linear_model import  ElasticNet
lel = ElasticNet(alpha=0.01)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
lel.fit(X_train,y_train)
yhat_lel=lel.predict(X_test)
#Turn continuous variables into discrete
ylist=[]
for output in yhat_lel:
  if output<4.5:
    output=round(output)
    ylist.append(output)
  else:
    output=4
    ylist.append(output)

y_hat_lel_rounded=np.array(ylist)
~~~
![image](https://user-images.githubusercontent.com/67920563/110249049-085fd780-7f42-11eb-9a36-fe203a3f56d3.png)

Our MAE becomes:
~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(lel.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_lel_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
![image](https://user-images.githubusercontent.com/67920563/110249094-5e347f80-7f42-11eb-8531-53bbd5cf94f8.png)

The MAE is about the same as our linear regression model at this alpha. Let us see the range of coefficients:
~~~
print('Coefficients:')
lel.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110249110-7a382100-7f42-11eb-9b1a-b4a308a85ef3.png)
~~~
min(lel.coef_)
~~~
![image](https://user-images.githubusercontent.com/67920563/110727329-023f5480-81e9-11eb-95b6-2deeb3d362cc.png)

~~~
max(lel.coef_)
~~~
![image](https://user-images.githubusercontent.com/67920563/110727339-08353580-81e9-11eb-8b44-3ae7b533842e.png)


Our range has shrunk to a difference of about 0.6. This is a great improvement from the non regularized model. We can also note that several features have coefficients of zero, meaning they have been deemed less necessary by the model.

~~~
list=[]
count=-1
counted_list=[]
A=lel.coef_
for a in A:
  count=count+1
  if a!=0:
    list.append(a)
    counted_list.append(count)
#Features
for c in counted_list:
  print(X.columns[c])
~~~
![image](https://user-images.githubusercontent.com/67920563/110249135-9c31a380-7f42-11eb-8440-9e0eadb7eae8.png)

Let us examine other alphas to see if there is a better MAE:
~~~
import matplotlib.pyplot as plt
maeEN=[]
for i in range(200):
  lel = ElasticNet(alpha=i)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
  lel.fit(X_train,y_train)
  yhat_lel=lel.predict(X_test)
  #Turn continuous variables into discrete
  ylist=[]
  for output in yhat_lel:
    if output<4.5:
      output=round(output)
      ylist.append(output)
    else:
      output=4
      ylist.append(output)

  y_hat_lel_rounded=np.array(ylist)
  maeEN.append(mean_absolute_error(y_test, y_hat_lel_rounded))
  plt.scatter(range(200),maeEN)
  ~~~
  ![image](https://user-images.githubusercontent.com/67920563/110249163-cedb9c00-7f42-11eb-88bf-7c4da7d4ba6e.png)
  
  From the graph we see that we can achieve a lower MAE once we increase the value of alpha. The graph plateaus at an MAE of less than 0.225.
  

## Standardized ElasticNet
Let us examine if standardizing the data can improve our results:
~~~
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
lel = ElasticNet(alpha=0.01)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
lel.fit(ss.fit_transform(X_train),y_train)
yhat_lel=lel.predict(ss.fit_transform(X_test))
#Turn continuous variables into discrete
ylist=[]
for output in yhat_lel:
  if output<4.5:
    output=round(output)
    ylist.append(output)
  else:
    output=4
    ylist.append(output)

y_hat_lel_rounded=np.array(ylist)
~~~
![image](https://user-images.githubusercontent.com/67920563/110249191-f92d5980-7f42-11eb-85a6-ad083f8a9fe1.png)
~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(lel.intercept_))
    
mae = mean_absolute_error(y_test, y_hat_lel_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
![image](https://user-images.githubusercontent.com/67920563/110249194-08aca280-7f43-11eb-8987-4eb281292927.png)

Our MAE at the same alpha is much lower than the unstandardized data. Let us examine our range of coefficients:

~~~
print('Coefficients:')
lel.coef_
~~~
![image](https://user-images.githubusercontent.com/67920563/110249211-1f52f980-7f43-11eb-9970-3ac4af8deb8f.png)
~~~
min(lel.coef_)
~~~
![image](https://user-images.githubusercontent.com/67920563/110684453-2ed47b80-81ab-11eb-9a76-798616ee3515.png)
~~~
max(lel.coef_)
~~~
![image](https://user-images.githubusercontent.com/67920563/110684433-28460400-81ab-11eb-9687-5744fd8e2e3f.png)

Our range of coefficients has shrunk by over 100%, going from a difference of ~0.6 to ~0.3

~~~
list=[]
count=-1
counted_list=[]
A=lel.coef_
for a in A:
  count=count+1
  if a!=0:
    list.append(a)
    counted_list.append(count)

#Features
for c in counted_list:
  print(X.columns[c])
~~~
![image](https://user-images.githubusercontent.com/67920563/110249268-6b9e3980-7f43-11eb-9295-b44d20bd82f6.png)
![image](https://user-images.githubusercontent.com/67920563/110249278-748f0b00-7f43-11eb-9398-d74e73279cff.png)

Examining other alphas, we note that greater alphas have no effect on the MAE:
~~~
import matplotlib.pyplot as plt
maeSEN=[]
for i in range(200):
  ss = StandardScaler()
  lel = ElasticNet(alpha=0.01)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)
  lel.fit(ss.fit_transform(X_train),y_train)
  yhat_lel=lel.predict(ss.fit_transform(X_test))
  #Turn continuous variables into discrete
  ylist=[]
  for output in yhat_lel:
    if output<4.5:
      output=round(output)
      ylist.append(output)
    else:
      output=4
      ylist.append(output)

  y_hat_lel_rounded=np.array(ylist)
  maeSEN.append(mean_absolute_error(y_test, y_hat_lel_rounded))
  plt.scatter(range(200),maeSEN)
~~~
![image](https://user-images.githubusercontent.com/67920563/110249314-99837e00-7f43-11eb-8cb5-7d88d106b469.png)

# Square Root Lasso/ Scaled Lasso

![image](https://user-images.githubusercontent.com/67920563/110669373-43f4de80-819a-11eb-8b7b-94e2d1c08b9e.png)

Square Root Lasso Regularization is a type of lasso that addresses noise in the data. A square root lasso model can fit potentially high-dimensional data. This type of model chooses an equilibrium with a sparse regression method by iteratively estimating the noise level via the mean residual square and the penalty in proportion to noise level.

## Application of Square Root Lasso in Python
Let us now try to apply Square Root Lasso in python. We can calculate the MAE using cross-validation:
~~~
! pip install --upgrade Cython
! pip install --upgrade git+https://github.com/statsmodels/statsmodels
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)

model = sm.OLS(y_train,X_train)
result = model.fit_regularized(method='sqrt_lasso', alpha=0.01)
yhat_test = result.predict(X_test)

  #Turn continuous variables into discrete
ylist=[]
for output in yhat_test:
  if output<0.5:
      output=1
      ylist.append(output)
  elif output<4.5:
      output=round(output)
      ylist.append(output)
  else:
      output=4
      ylist.append(output)

y_hat_test_rounded=np.array(ylist)
y_hat_test_rounded
~~~
![image](https://user-images.githubusercontent.com/67920563/110687730-d7380f00-81ae-11eb-9629-925199207f3b.png)

Let us calculate our MAE:
~~~
MAE(y_test,y_hat_test_rounded)
~~~
![image](https://user-images.githubusercontent.com/67920563/110687886-fa62be80-81ae-11eb-9e26-4ee084ad7867.png)

Our MAE is 0.34 at alpha 0.01, which is on the higher end of our models. Let us look at the range of the coefficients:
~~~
result.params
~~~
![image](https://user-images.githubusercontent.com/67920563/110674528-fa0ef700-819f-11eb-8477-de0c7f4ffa9e.png)
~~~
min(result.params)
~~~
![image](https://user-images.githubusercontent.com/67920563/110684312-fe8cdd00-81aa-11eb-8e02-4b308341c58f.png)
~~~
max(result.params)
~~~
![image](https://user-images.githubusercontent.com/67920563/110684328-051b5480-81ab-11eb-88fc-b703ac8321bf.png)

We can see from the results that the range of coefficients has increased; this type of regularization may not be adequate for our model.

Let us examine other alpha values to see if we can improve our results:
~~~
maeSL=[]
for i in range(200):
  model = sm.OLS(y_train,X_train)
  result = model.fit_regularized(method='sqrt_lasso', alpha=i)
  yhat_test = result.predict(X_test)
  maeSL.append(MAE(y_test,yhat_test))
 plt.scatter(range(200),maeSL)
 ~~~
![image](https://user-images.githubusercontent.com/67920563/110690092-81189b00-81b1-11eb-863f-81c264065315.png)


Our results show that the MAE is never better than 0.340, and increases as alpha is increased.
  
## Scaled Square Root Lasso
Let us try to improve our model by scaling our data:
~~~
scale = StandardScaler()
Xs_train = scale.fit_transform(X_train)
Xs_test  = scale.transform(X_test)
model = sm.OLS(y_train,Xs_train)
result = model.fit_regularized(method='sqrt_lasso', alpha=0.5)
yhat_test = result.predict(Xs_test)
#Turn continuous variables into discrete
ylist=[]
for output in yhat_test:
  if output<0.5:
      output=1
      ylist.append(output)
  elif output<4.5:
      output=round(output)
      ylist.append(output)
  else:
      output=4
      ylist.append(output)

y_hat_test_rounded=np.array(ylist)
y_hat_test_rounded
~~~
![image](https://user-images.githubusercontent.com/67920563/110688273-647b6380-81af-11eb-9773-64a4021e02ff.png)

Our predicted values look very different than our actual values. Let us look at our MAE:
~~~
MAE(y_test,y_hat_test_rounded)
~~~
![image](https://user-images.githubusercontent.com/67920563/110688387-84128c00-81af-11eb-9076-7b76ed6eff2c.png)

Our MAE is now extremely high. Let us look at the range of our coefficients.

~~~
result.params
~~~
![image](https://user-images.githubusercontent.com/67920563/110685644-7f000d80-81ac-11eb-8916-36774aed78af.png)

~~~
min(result.params)
~~~
![image](https://user-images.githubusercontent.com/67920563/110685400-38121800-81ac-11eb-8753-71be4400edc9.png)
~~~
max(result.params)
~~~
![image](https://user-images.githubusercontent.com/67920563/110685474-511ac900-81ac-11eb-93f3-914183b4f28d.png)

Our range of coefficients is not high, but due to our large MAE, this model is not adequate for our data.
Let us examine other alpha values:
~~~
maeSSL=[]
for i in range(200):
  model = sm.OLS(y_train,Xs_train)
  result = model.fit_regularized(method='sqrt_lasso', alpha=0.5)
  yhat_test = result.predict(Xs_test)
  maeSSL.append(MAE(y_test,yhat_test))

plt.scatter(range(200),maeSSL)
~~~
![image](https://user-images.githubusercontent.com/67920563/110688949-23d01a00-81b0-11eb-974f-379112c17dfd.png)


As we can see, increasing or decreasing the alpha values did not have an impact on our MAE.

# SCAD
The last feature selection method we have is Smoothly Clipped Absolute Deviations (SCAD). 

![image](https://user-images.githubusercontent.com/67920563/110722689-b7214380-81e0-11eb-9864-c6611ed70b7a.png)


SCAD suffers from bias but encourages sparsity, thus allowing for larger weights. This type of penalty relaxes the rate of penalization as the absolute value of the weight coefficient increases, unlike Lasso regularization which increases the penalty with respect to the weight.

## Application of SCAD to IDB problem
We begin by reshaping the data and defining the SCAD algorithm:
~~~
y=y.values.reshape(-1,1)
X=X.values

from scipy.optimize import minimize
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
    
def scad(beta):
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  n = len(y)
  return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))
  
def dscad(beta):
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  n = len(y)
  return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,lam,a)).flatten()
  
p = X.shape[1]
b0 = np.random.normal(1,1,p)

lam = 0.5
a = 0.001
output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
yhat_test_scad = X_test.dot(output.x)

output.x
~~~
![image](https://user-images.githubusercontent.com/67920563/110725703-061da780-81e6-11eb-9c9e-c5ccc8c4bd61.png)

~~~
min(output.x)
~~~
![image](https://user-images.githubusercontent.com/67920563/110725716-0d44b580-81e6-11eb-9b30-45808ce4c4d4.png)

~~~
max(output.x)
~~~
![image](https://user-images.githubusercontent.com/67920563/110725744-15045a00-81e6-11eb-874e-2f7eb9d651bb.png)


The range of our output is rather large, with a difference of ~.
~~~
yhat_test_scad = X_test.dot(output.x)
yhat_test_scad
~~~
![image](https://user-images.githubusercontent.com/67920563/110725764-1fbeef00-81e6-11eb-8a3a-8893ff44f40b.png)


Let us find the predicted y values:
~~~
ylist=[]
for output in yhat_test_scad:
  if output<0.5:
      output=1
      ylist.append(output)
  elif output<4.5:
      output=round(output)
      ylist.append(output)
  else:
      output=4
      ylist.append(output)

y_hat_test_rounded=np.array(ylist)
y_hat_test_rounded
~~~
![image](https://user-images.githubusercontent.com/67920563/110725805-349b8280-81e6-11eb-96a3-5827a36e23ef.png)


Let us find the MAE at this beta:
~~~
mae = mean_absolute_error(y_test, y_hat_test_rounded)
print("MAE = {:,.2f}".format(mae))
~~~
![image](https://user-images.githubusercontent.com/67920563/110725830-40874480-81e6-11eb-9451-34675c4956b7.png)



Our MAE is extremely high for our data, as it predicted a lot of 1 values. 
Let us try to use other alpha values:
~~~
listofoutput=[]
lam = 0.5
for i in range(200):
  a = i
  output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
  yhat_test_scad = X_test.dot(output.x)
  ylist=[]
  for output in yhat_test_scad:
    if output<0.5:
        output=1
        ylist.append(output)
    elif output<4.5:
        output=round(output)
        ylist.append(output)
    else:
        output=4
        ylist.append(output)

  y_hat_test_rounded=np.array(ylist)
  listofoutput.append(mean_absolute_error(y_test, y_hat_test_rounded))
  
plt.scatter(range(200),listofoutput)
~~~
![image](https://user-images.githubusercontent.com/67920563/110724745-5eec4080-81e4-11eb-9975-ab8e2322f761.png)
~~~
min(listofoutput)
~~~
![image](https://user-images.githubusercontent.com/67920563/110724767-69a6d580-81e4-11eb-981d-398a12991882.png)

We can see that the minimum MAE is 0.21 and occurs at some alpha in the range of 0 to 200

## Standardized SCAD with the IDB dataset
Let us see if standardizing our dataset can help lower our MAE:
~~~
y=y.values.reshape(-1,1)
X=X.values

Xs = scale.fit_transform(X)
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
    

def scad(beta):
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  n = len(y)
  return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))
  
def dscad(beta):
  beta = beta.flatten()
  beta = beta.reshape(-1,1)
  n = len(y)
  return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,lam,a)).flatten()
  
p = Xs.shape[1]
b0 = np.random.normal(1,1,p)

lam = 1
a = 0.01
output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})

output.x
~~~
![image](https://user-images.githubusercontent.com/67920563/110726403-521d1c00-81e7-11eb-8297-c37b33557cfa.png)
Our Min and Max Output:
~~~
min(output.x)
~~~
![image](https://user-images.githubusercontent.com/67920563/110726445-6234fb80-81e7-11eb-8f9e-62909e60fab4.png)
~~~
max(output.x)
~~~
![image](https://user-images.githubusercontent.com/67920563/110726460-69f4a000-81e7-11eb-85d5-380203cd1d0b.png)

Our y Predictions:
~~~
yhat_test_scad = X_test.dot(output.x)
yhat_test_scad
~~~

![image](https://user-images.githubusercontent.com/67920563/110726491-7aa51600-81e7-11eb-9fdc-cf22f6fd72f1.png)


~~~
ylist=[]
for output in yhat_test_scad:
  if output<0.5:
      output=1
      ylist.append(output)
  elif output<4.5:
      output=round(output)
      ylist.append(output)
  else:
      output=4
      ylist.append(output)

y_hat_test_rounded=np.array(ylist)
y_hat_test_rounded
~~~

![image](https://user-images.githubusercontent.com/67920563/110726522-8690d800-81e7-11eb-8118-52534aad4cf2.png)

And finally...our MAE
~~~
mae = mean_absolute_error(y_test, y_hat_test_rounded)
print("MAE = {:,.2f}".format(mae))
~~~

![image](https://user-images.githubusercontent.com/67920563/110726547-91e40380-81e7-11eb-9e88-2b935dae0727.png)
As we can see, the MAE was hardly reduced and still remains above the majority of the other models. Let us examine if it can be improved at other alphas:

~~~
listofoutput=[]
lam = 0.5
for i in range(200):
  a = i
  output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
  yhat_test_scad = X_test.dot(output.x)
  ylist=[]
  for output in yhat_test_scad:
    if output<0.5:
        output=1
        ylist.append(output)
    elif output<4.5:
        output=round(output)
        ylist.append(output)
    else:
        output=4
        ylist.append(output)

  y_hat_test_rounded=np.array(ylist)
  listofoutput.append(mean_absolute_error(y_test, y_hat_test_rounded))
  
plt.scatter(range(200),listofoutput)
~~~
![image](https://user-images.githubusercontent.com/67920563/110726603-a922f100-81e7-11eb-8680-f2ca4a658b0f.png)

~~~
min(listofoutput)
~~~

![image](https://user-images.githubusercontent.com/67920563/110726626-b4761c80-81e7-11eb-838a-254604095c9b.png)

Our minimum MAE is close to our unstandardized SCAD MAE.

## Comparison of Regularization Techniques

![image](https://user-images.githubusercontent.com/67920563/110728267-bb525e80-81ea-11eb-823e-5faeccfc886f.png)


After trying four different regularization techniques and trying both standardized and unstandardized versions of our models, we can conclude that the Standardized Ridge Regularization reduced our MAE the most (0.15).
Looking further at the data table, we can note that Lasso Regularization shrunk the model the most in terms of the size of the weights (~0.3 between the max and min coefficient). It is not a surprise that Elastic Net (~0.6) comes in a close second in respect to this, as it is partially made of Lasso Regularization. This confirms that Lasso Regularization will throw out the terms it does not need and will simplify the model in an extreme way. It can also be confirmed that Lasso Regularization was able to decently reduce the MAE because our data did not have strong correlations between its features. Had the data had stronger correlatations, we may have seen a larger difference between Lasso and Ridge. Because Ridge outperformed Lasso by a little in this exercise, we may assume that Lasso may have slightly oversimplified the relationship between features. However, with so many features and such little difference between Lasso and Ridge Regularization, it may be better to opt for Lasso Regularization due to its feature selection capabilities.
We can also note that at the selected alpha (a=0.01), Standardized Lasso performed best(0.23), while the other regularization techniques were scattered. It can also be noted that SCAD worked best at greater alphas, and although at a=0.01 it had a quite large MAE, its minimum MAE was in range. Although all the regularization techniques lowered the MAE into the 0.2s, standardized square root lasso did not improve at all. 

# Significant Earthquakes, 1965-2016
The same regularization analysis can be tested on different datasets. This dataset describes major earthquakes greater than a magnitude of 5.5 from 1965 to 2016 and can be found on Kaggle. We will attempt to predict earthquake magnitude from factors such as latitude, length, and depth. 

~~~
#import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv('/content/drive/MyDrive/database.csv')
df2=df
df2=df2.drop('Status', axis=1)
df2=df2.drop('Date', axis=1)
df2=df2.drop(['Time','Type','ID', 'Source', 'Location Source', 'Magnitude Source'], axis=1)
df2=df2.drop('Magnitude Type', axis=1)
df2=df2.apply(pd.to_numeric)

mean_value1=df2['Depth Error'].mean()
df2['Depth Error'].fillna(mean_value1, inplace=True)
mean_value2=df2['Depth Seismic Stations'].mean()
df2['Depth Seismic Stations'].fillna(mean_value2, inplace=True)
mean_value3=df2['Magnitude Error'].mean()
df2['Magnitude Error'].fillna(mean_value3, inplace=True)
mean_value4=df2['Magnitude Seismic Stations'].mean()
df2['Magnitude Seismic Stations'].fillna(mean_value4, inplace=True)
mean_value5=df2['Azimuthal Gap'].mean()
df2['Azimuthal Gap'].fillna(mean_value5, inplace=True)
mean_value6=df2['Horizontal Distance'].mean()
df2['Horizontal Distance'].fillna(mean_value6, inplace=True)
mean_value7=df2['Horizontal Error'].mean()
df2['Horizontal Error'].fillna(mean_value7, inplace=True)
df2=df2.drop('Root Mean Square', axis=1)
~~~

![image](https://user-images.githubusercontent.com/67920563/111014396-7442a580-8371-11eb-8d0f-ef637251c388.png)














