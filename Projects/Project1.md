#Project 0

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



This particular project will explore the relationship between rooms and housing prices.

## Visualizing the Data
