# Project 1

Comparing the performance of different regressors with the goal of predicting housing prices from the number of rooms.
Dataset Used: Boston Housing Data

## Introducing the Data
The Boston Housing dataset is a collection from the U.S. Census Service concerning Boston, MA's housing. The dataset has 506 total data points and 17 attributes. The Dataset's attributes are listed as follows:
1. crime - per capita crime rate by town
2. residential - proportion of residential land zoned for lots over 25,000 sq.ft.
3. industrial - proportion of non-retail business acres per town.
4. river - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. nox - nitric oxides concentration (parts per 10 million)
6. rooms - average number of rooms per dwelling
7. older - proportion of owner-occupied units built prior to 1940
8. distance - weighted distances to five Boston employment centres
9. highway - index of accessibility to radial highways
10. tax - full-value property-tax rate per $10,000
11. ptratio - pupil-teacher ratio by town
12. town
13. lstat - % lower status of the population
14. cmedv - Median value of owner-occupied homes in $1000's
15. tract - year
16. longitude
17. latitude

### Establishing a Data Frame
In order to access the data in Colab, we must first upload the dataset to google drive and then mount the drive on Colab:
~~~
from google.colab import drive
drive.mount('/content/drive')
~~~
Once the data is accessible, we must import the libraries needed to create a data frame:
~~~
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
~~~
Then, we create the data frame. If we do not need to clean it up or make any adjustments, we can move on to the next step:
~~~
df = pd.read_csv('drive/MyDrive/BostonHousingPrices.csv')
~~~

INSERT PICTURE OF DF

This particular project will explore the relationship between rooms and housing prices.

## Visualizing the Data
In order to get a better perspective on what type of regression may be best for our model, let's plot out the two attributes:
~~~
%matplotlib inline
%config InlineBackend.figure_format = 'retina' #Formats the visual representation
fig, ax = plt.subplots(figsize=(10,8)) #handles the size of plot

ax.set_xlim(3, 9) #boundaries of x
ax.set_ylim(0, 51) #boundaries of y

ax.scatter(x=df['rooms'], y=df['cmedv'],s=25) #plots the points

ax.set_xlabel('Number of Rooms',fontsize=16,color='darkgreen') #labels for x
ax.set_ylabel('House Price (Thousands of Dollars)',fontsize=16,color='darkgreen') #labels for y
ax.set_title('Boston Housing Prices',fontsize=18,color='purple') #plot label

ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8) #sets grid
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
plt.show()
~~~
INSERT PICTURE OF PLOT
The plot seems to show at first glance a positive increasing linear trend. However, we also note that there are large clusters and a lot of noise, possibly making this a difficult dataset to model.

## Prediction Modeling
Prediction Modeling in Machine Learning requires one to split the data into a training set and testing set. The parameters chosen are largely dependent on the problem at hand and the data set itself. There are many types of prediction models we can apply to data, some being more accurate than others. One way to measure a model's accuracy is through the MAE or Mean Absolute Error. This error metric is essentially the sum of all absolute errors in the model's predictions. Absolute Errors take the absolute value of the difference between an actual value and the predicted value. On a plot, we can measure the MAE as the average vertical distance between each point and the predicted line. The smaller the error, the better the predictive model.

INSERT PICTURE MAE

## Linear Regression Predictive Model
Our problem type deems it necessary to derive a continuous numerical value as our predicted value. The nature of our problem thus makes a linear regression algorithm a great candidate for our predictive model. Linear Regressions establish a linear relationship between and independent variable x and dependent variable y. Linear Regressions find the line of best fit in the data and use this to predict future values. In order to find the line of best fit, optimization algorithms such as gradient descent are used. Outliers in the data may cause overfitting of the linear regression, thus lowering the overal testing accuracy.

Linear Regression models are statistical models, but how do they differ from statistical analysis? While in statistics we calculate the unknowns using statistical formulas, linear regression in machine learning only uses optimization algorithms. For example, assuming we have several dimensions of variables, statistical analysis will tell us to find our beta values (correlation coefficients) and any constants or intercepts. Machine learning will find these values by simply minimizing error.

Linear Regression not only helps one predict variables but also quantify the impact independent variables have on the dependent variable as well as identifying the most important variables.

### Modeling Linear Regressions in Python
In order to begin with our model, we must first indicate our independent and dependent variables and seperate the data into training and testing sets.
~~~
from sklearn.model_selection import train_test_split
X = np.array(df['rooms']).reshape(-1,1)
y = np.array(df['cmedv']).reshape(-1,1)
~~~
We will choose a test size of 0.3 and random state 2021 in order to have consistency across all models. 
~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021)

y_train = y_train.reshape(len(y_train),)
y_test = y_test.reshape(len(y_test),)
~~~
Lastly, we will apply the linear regression model from the sklearn library.
~~~
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
~~~

### Analyzing our Model
As mentioned earlier, we can use mean absolute error in order to determine the accuracy of our model. We can import the mean absolute error package from sklearn to determine the deviation from the actual value and predicted value. 
~~~
from sklearn.metrics import mean_absolute_error 
print("Intercept: {:,.3f}".format(lm.intercept_))
print("Coefficient: {:,.3f}".format(lm.coef_[0]))
    
mae = mean_absolute_error(y_test, lm.predict(X_test))
print("MAE = ${:,.2f}".format(1000*mae))
~~~
We can also plot our linear regression:
~~~
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(3, 9)
ax.set_ylim(0, 51)
ax.scatter(x=df['rooms'], y=df['cmedv'],s=25)
ax.plot(X_test, lm.predict(X_test), color='red') #linear regression
ax.set_xlabel('Number of Rooms',fontsize=16,color='Darkgreen')
ax.set_ylabel('House Price (Tens of Thousands of Dollars)',fontsize=16,color='Darkgreen')
ax.set_title('Boston Housing Prices',fontsize=16,color='purple')
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
~~~
INSERT PICTURE OF LINEAR REGRESSION
The Model has an MAE of $4,109.44. Our linear regression model has a rather high MAE, thus making this a weak learner. Because of the clustering of data points in an unlinear fashion, our model may have lost accuracy trying to fit such points. 

## Kernel Weighted Local Regression
LOWESS or locally weighted linear regressions are non-parametric regressions in which the simplicity of a linear regression is combined with the flexibility of non linear regression. LOWESS regressions are used when there are non-linear relationships between variables. In essence, linear functions using weighted least squares are fitted only on local sets of data, thus building up to a function that describes the overall variation. For each value of x, the value of f(x) is estimated using its neighboring known values. 

Let us say we choose a specific data set point X1 for which we want to predict its Y1 value. We can determine its neighbors by choosing a specific distance which will result in some ordered set A. This set will be converted into another weighted set using a weight/kernel function. The specific weights will depend on what kernel function is chosen. 
Example: Below we have chosen a tri-cubic function as our weight function. 

INSERT PICTURE OF FUNCTION

The heights of the kernel function determine the weights at specific points. The neighbors closer to the target point X1 will have higher weights than points further away. In other words, by locally weighing points, we can assign higher importance to training data that is closest to the target point. 

For every target point X1, LOWESS will apply a linear regression that will calculate the corresponding Y1 value. Although this algorithm may work well for regression applications with complex deterministic structures, LOWESS has several disadvantages. Because a model is computed for each point, it is very computationally intensive as well as having a large parameter size. It is still quite volatile to outliers in the data set as well as it cannot be translated into a mathematical formula.





