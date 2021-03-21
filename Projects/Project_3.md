Nonlinear Multivariate Regression Methods

# Pre-Processing
~~~
#Mount drive
from google.colab import drive
drive.mount('/content/drive')

#import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv('drive/MyDrive/BostonHousingPrices.csv')
~~~
![image](https://user-images.githubusercontent.com/67920563/111891365-eb5ae800-89c8-11eb-86ff-c48d6169890b.png)
~~~
#PREDICTION-split the data into training and testing
from sklearn.model_selection import train_test_split

X=df.drop(['cmedv', 'town'], axis=1)
y = np.array(df['cmedv']).reshape(-1,1)

corr_matrix=np.corrcoef(X)
corr_matrix
~~~
![image](https://user-images.githubusercontent.com/67920563/111891404-55738d00-89c9-11eb-8a7c-ace78b230a5f.png)
~~~
import seaborn as sns
plt.figure(figsize=(20,20))
sns.heatmap(X.corr(), vmax=1, vmin=-1, cmap="spring", annot=True,fmt='.2f')
plt.show()
~~~
![image](https://user-images.githubusercontent.com/67920563/111891413-658b6c80-89c9-11eb-9177-a62d5cc46d9a.png)
~~~
sns.pairplot(data=df, kind='scatter', hue='cmedv')
plt.show()
~~~
![image](https://user-images.githubusercontent.com/67920563/111891428-9075c080-89c9-11eb-8ce8-1710086928aa.png)
~~~
from scipy import stats
def tstat_corr(r,n):
  t = r*np.sqrt(n-2)/np.sqrt(1-r**2)
  pval = stats.t.sf(np.abs(t), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
  print('t-statistic = %6.3f pvalue = %6.4f' % (t, pval))
  
tstat_corr(0.06,len(X)) #put in an r value from matrix
~~~
# Nonlinear Lasso Regularization
~~~
features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
X = np.array(df[features])
y = np.array(df['cmedv']).reshape(-1,1)
Xdf = df[features]

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import statsmodels.api as sm

scale = StandardScaler()
poly = PolynomialFeatures(degree=3)

def DoKFold_SK(X,y,model,k):
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  pipe = Pipeline([('scale',scale),('polynomial features',poly),('model',model)])
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    pipe.fit(X_train,y_train)
    yhat_test = pipe.predict(X_test)
    PE.append(MAE(y_test,yhat_test))
  return 1000*np.mean(PE)
~~~
  
## Testing different K-Fold Numbers 
~~~
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
model = Lasso(alpha=0.05,max_iter=5000)
DoKFold_SK(X,y,model,10)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891504-f5311b00-89c9-11eb-9b4b-0f1575d65ff0.png)
~~~
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
model = Lasso(alpha=0.05,max_iter=5000)
DoKFold_SK(X,y,model,30)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891687-0e869700-89cb-11eb-80b7-d4c3eda3a2dd.png)
~~~
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
model = Lasso(alpha=0.05,max_iter=5000)
DoKFold_SK(X,y,model,300)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891691-16463b80-89cb-11eb-92cf-cfc707692281.png)
~~~
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
model = Lasso(alpha=0.05,max_iter=5000)
DoKFold_SK(X,y,model,506)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891697-1e05e000-89cb-11eb-9a2f-bda2479e338e.png)

## Testing Different Alphas
~~~
#test alphas
a_range= np.linspace(0.01, 100)
test_mae=[]
for a in a_range:
  test_mae.append(DoKFold_SK(X,y,model,10))
  
min(test_mae)  
~~~
![image](https://user-images.githubusercontent.com/67920563/111891520-12fe8000-89ca-11eb-9b6c-fc1ba79c0443.png)
~~~
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
ax.scatter(a_range, test_mae)
ax.plot(a_range, test_mae, c='red')
~~~
![image](https://user-images.githubusercontent.com/67920563/111891525-21e53280-89ca-11eb-8b50-969fdd24e7df.png)

## Testing Different Polynomial Numbers
- At degree=4, k=10, alpha=0.05: 2334.030807616874
- At degree=4, k=30, alpha=0.05: 2263.9195401639035
- At degree=4, k=300, alpha=0.05: 2249.1800237285274
- At degree=4, k=506, alpha=0.05: 2260.300028069687


# Nonlinear Ridge Regression
## Testing Different K -Fold Values
~~~
model = Ridge(alpha=0.05)
DoKFold_SK(X,y,model,10)
~~~
![image](https://user-images.githubusercontent.com/67920563/111896130-c167ec80-89ed-11eb-9f71-3da401872699.png)

~~~
model = Ridge(alpha=0.05)
DoKFold_SK(X,y,model,30)
~~~
![image](https://user-images.githubusercontent.com/67920563/111896134-caf15480-89ed-11eb-9972-2be560e847e0.png)

~~~
model = Ridge(alpha=0.05)
DoKFold_SK(X,y,model,300)
~~~
![image](https://user-images.githubusercontent.com/67920563/111896140-d2b0f900-89ed-11eb-9606-bff20d290abe.png)

~~~
model = Ridge(alpha=0.05)
DoKFold_SK(X,y,model,506)
~~~
![image](https://user-images.githubusercontent.com/67920563/111896148-dba1ca80-89ed-11eb-8ce2-a9f4fa019d63.png)


## Testing Different Alphas
~~~
#test alphas
a_range= np.linspace(0.01, 100)
test_mae=[]
for a in a_range:
  test_mae.append(DoKFold_SK(X,y,model,10))
min(test_mae)  
~~~  
![image](https://user-images.githubusercontent.com/67920563/111896196-173c9480-89ee-11eb-9339-e4f078e2ef2c.png)

~~~
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
ax.scatter(a_range, test_mae)
ax.plot(a_range, test_mae, c='red')
~~~
![image](https://user-images.githubusercontent.com/67920563/111896205-1efc3900-89ee-11eb-9dd1-1a7fab807190.png)


## Testing Different Polynomial Degrees

- At degree=4, k=10, alpha=0.05: 6763.16199953988
- At degree=4, k=30, alpha=0.05: 6722.900116892813
- At degree=4, k=300, alpha=0.05: 8321.568943584169
- At degree=4, k=506, alpha=0.05: 6847.485429539462

# Elastic Net
~~~
model = ElasticNet(alpha=0.05,l1_ratio=0.25,max_iter=12000)
DoKFold_SK(X,y,model,10)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891925-029bd480-89cd-11eb-88ed-47b1d2adcdc4.png)
~~~
model = ElasticNet(alpha=0.05,l1_ratio=0.25,max_iter=12000)
DoKFold_SK(X,y,model,30)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891958-42fb5280-89cd-11eb-816c-df825e530b2f.png)
~~~
model = ElasticNet(alpha=0.05,l1_ratio=0.25,max_iter=12000)
DoKFold_SK(X,y,model,300)
~~~
![image](https://user-images.githubusercontent.com/67920563/111892031-eea4a280-89cd-11eb-9173-c57c7bad8d40.png)

~~~
model = ElasticNet(alpha=0.05,l1_ratio=0.25,max_iter=12000)
DoKFold_SK(X,y,model,506)
~~~
![image](https://user-images.githubusercontent.com/67920563/111892101-94f0a800-89ce-11eb-9d79-e69d0cac97e1.png)


## Testing Alphas
~~~
#test alphas
a_range= np.linspace(0.01, 100)
test_mae=[]
for a in a_range:
  test_mae.append(DoKFold_SK(X,y,model,10))
  
min(test_mae)
~~~
![image](https://user-images.githubusercontent.com/67920563/111892105-9ae68900-89ce-11eb-8b5b-afd614f2b29a.png)

~~~
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
ax.scatter(a_range, test_mae)
ax.plot(a_range, test_mae, c='red')
~~~
![image](https://user-images.githubusercontent.com/67920563/111892001-a38a8f80-89cd-11eb-9d68-496ca06edd72.png)

# Squart Root Lasso Regression
~~~
scale = StandardScaler()
def sqrtlasso_model(X,y,alpha):
  n = X.shape[0]
  p = X.shape[1]
  #we add an extra columns of 1 for the intercept
  #X = np.c_[np.ones((n,1)),X]
  def sqrtlasso(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.sqrt(1/n*np.sum((y-X.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
  
  def dsqrtlasso(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.array((-1/np.sqrt(n))*np.transpose(X).dot(y-X.dot(beta))/np.sqrt(np.sum((y-X.dot(beta))**2))+alpha*np.sign(beta)).flatten()
  b0 = np.ones((p,1))
  output = minimize(sqrtlasso, b0, method='L-BFGS-B', jac=dsqrtlasso,options={'gtol': 1e-8, 'maxiter': 1e8,'maxls': 25,'disp': True})
  return output.x
  
def DoKFoldSqrt(X,y,a,k,d):
  PE = []
  scale = StandardScaler()
  poly = PolynomialFeatures(degree=d)
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    X_train_scaled = scale.fit_transform(X_train)
    X_train_poly = poly.fit_transform(X_train_scaled)
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    X_test_scaled = scale.transform(X_test)
    X_test_poly = poly.fit_transform(X_test_scaled)
    y_test  = y[idxtest]
    beta_sqrt = sqrtlasso_model(X_train_poly,y_train,a)
    n = X_test_poly.shape[0]
    p = X_test_poly.shape[1]
    # we add an extra columns of 1 for the intercept
    #X1_test = np.c_[np.ones((n,1)),X_test]
    yhat_sqrt = X_test_poly.dot(beta_sqrt)
    PE.append(MAE(y_test,yhat_sqrt))
  return 1000*np.mean(PE)
~~~ 

## Testing Different K-Fold Values
~~~  
DoKFoldSqrt(X,y,0.05,10,3)
~~~
![image](https://user-images.githubusercontent.com/67920563/111894029-bb6a0f80-89dd-11eb-8a1a-368483a85800.png)

~~~
DoKFoldSqrt(X,y,0.05,30,3)
~~~

![image](https://user-images.githubusercontent.com/67920563/111894035-c3c24a80-89dd-11eb-9ecd-263dd8744d8a.png)

~~~
DoKFoldSqrt(X,y,0.05,300,3)
~~~

![image](https://user-images.githubusercontent.com/67920563/111895199-6c28dc80-89e7-11eb-97de-78498cd8b9aa.png)

~~~
DoKFoldSqrt(X,y,0.05,506,3)
~~~
![image](https://user-images.githubusercontent.com/67920563/111895205-71862700-89e7-11eb-8a06-948cc2794574.png)



## Test Alphas
~~~
#test alphas
a_range= np.linspace(0.01, 10,100)
test_mae=[]
for a in a_range:
  test_mae.append(DoKFoldSqrt(X,y,a,10,3))

min(test_mae)
~~~  
![image](https://user-images.githubusercontent.com/67920563/111895461-0b020880-89e9-11eb-9c06-9494d0744e5c.png)

~~~
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
ax.scatter(a_range, test_mae)
ax.plot(a_range, test_mae, c='red')
~~~
![image](https://user-images.githubusercontent.com/67920563/111895457-076e8180-89e9-11eb-8da2-da9b7742e371.png)


## Testing Polynomial Degrees

~~~
DoKFoldSqrt(X,y,0.05,10,2)
~~~
![image](https://user-images.githubusercontent.com/67920563/111894089-44814680-89de-11eb-8fd4-189b08e9f499.png)

~~~
DoKFoldSqrt(X,y,0.05,10,4)
~~~
![image](https://user-images.githubusercontent.com/67920563/111895210-7ba82580-89e7-11eb-9d72-901530d434d2.png)


# SCAD
~~~
from numba import jit, prange
from scipy.optimize import minimize

@jit
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

def scad_model(X,y,lam,a):
  n = X.shape[0]
  p = X.shape[1]
  # we add aan extra columns of 1 for the intercept
 # X = np.c_[np.ones((n,1)),X]
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
  b0 = np.ones((p,1))
  output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 1e7,'maxls': 25,'disp': True})
  return output.x
  
def DoKFoldScad(X,y,lam,a,k):
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  #pipe2 = Pipeline([('scale',scale),('polynomial features',poly),('model',model)])
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    X_train_scaled=scale.fit_transform(X_train)
    X_train_poly=poly.fit_transform(X_train_scaled)

    y_train = y[idxtrain]

    X_test  = X[idxtest,:]
    X_test_scaled=scale.transform(X_test)
    X_test_poly=poly.fit_transform(X_test_scaled)

    y_test  = y[idxtest]
    #pipe.fit(X_train, y_train)

    beta_scad = scad_model(X_train_poly,y_train,lam,a)
    n = X_test_poly.shape[0]
    p = X_test_poly.shape[1]
    # we add an extra columns of 1 for the intercept
    #X1_test = np.c_[np.ones((n,1)),X_test]
    #yhat_scad=pipe.predict(beta_scad)
    yhat_scad = X_test_poly.dot(beta_scad)
    PE.append(MAE(y_test,yhat_scad))
  return 1000*np.mean(PE)  
~~~
## Testing K-Fold Values
~~~
DoKFoldScad(X,y,0.05,0.1, 10)
~~~
![image](https://user-images.githubusercontent.com/67920563/111892418-c159f380-89d1-11eb-917e-f5301b1deefa.png)
~~~
DoKFoldScad(X,y,0.05,0.1, 30)
~~~
![image](https://user-images.githubusercontent.com/67920563/111892423-c7e86b00-89d1-11eb-9728-6acb7a985bdf.png)
~~~
DoKFoldScad(X,y,0.05,0.1, 300)
~~~
![image](https://user-images.githubusercontent.com/67920563/111892429-ce76e280-89d1-11eb-8622-d73910c974e9.png)
~~~
DoKFoldScad(X,y,0.05,0.1, 506)
~~~
![image](https://user-images.githubusercontent.com/67920563/111892430-d6368700-89d1-11eb-9324-84d72360585d.png)
  
## Testing Alphas
~~~
#test alphas
a_range= np.linspace(0.01, 100)
test_mae=[]
for a in a_range:
  test_mae.append(DoKFoldScad(X,y,0.05,a,10))
  
 min(test_mae)
 ~~~
![image](https://user-images.githubusercontent.com/67920563/111892498-64ab0880-89d2-11eb-9f2d-d3f96082f9d4.png)
 
 ~~~
 import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
ax.scatter(a_range, test_mae)
ax.plot(a_range, test_mae, c='red')
~~~
![image](https://user-images.githubusercontent.com/67920563/111892500-6aa0e980-89d2-11eb-824a-f05b6a07b866.png)

 

# Neural Networks
~~~
dat = np.concatenate([X,y], axis=1)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split as tts
dat_train, dat_test = tts(dat, test_size=0.3, random_state=1234)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

dat_train[:,:-1].shape

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Create a Neural Network model
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=11))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
# Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
# Typically ReLu-based activation are used but since it is performed regression, it is needed a linear activation.
model.add(Dense(1, activation="relu"))

# Compile model: The model is initialized with the Adam optimizer and then it is compiled.
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=800)

# Fit the model
history = model.fit(dat_train[:,:-1], dat_train[:,11], validation_split=0.3, epochs=1000, batch_size=100, verbose=0, callbacks=[es])

from sklearn.metrics import mean_absolute_error

yhat_nn = model.predict(dat_test[:,:-1])
mae_nn = mean_absolute_error(dat_test[:,-1], yhat_nn)
print("MAE Neural Network = ${:,.2f}".format(1000*mae_nn))
~~~
![image](https://user-images.githubusercontent.com/67920563/111892725-5231ce80-89d4-11eb-892a-5b2fdc76f079.png)
~~~
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=1693)

#%%timeit -n 1

mae_nn = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0:-1]
  y_train = dat[idxtrain,-1]
  X_test  = dat[idxtest,0:-1]
  y_test = dat[idxtest,-1]
  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=800)
  model.fit(X_train,y_train,validation_split=0.3, epochs=100, batch_size=100, verbose=0, callbacks=[es])
  yhat_nn = model.predict(X_test)
  mae_nn.append(mean_absolute_error(y_test, yhat_nn))
print("Validated MAE Neural Network Regression = ${:,.2f}".format(1000*np.mean(mae_nn)))
~~~
![image](https://user-images.githubusercontent.com/67920563/111892757-9c1ab480-89d4-11eb-9c67-a57849aa0515.png)

At k=30:

![image](https://user-images.githubusercontent.com/67920563/111892830-2b27cc80-89d5-11eb-909d-c46923e49556.png)

At k=300:

![image](https://user-images.githubusercontent.com/67920563/111893313-41835780-89d8-11eb-8c70-6d22142de5f4.png)

At k=506:

![image](https://user-images.githubusercontent.com/67920563/111893932-f15ac400-89dc-11eb-982b-def94c3107ea.png)


# XGBOOST
~~~
import xgboost as xgb

model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=1,alpha=1,gamma=1,max_depth=10)

model_xgb.fit(dat_train[:,:-1],dat_train[:,-1])
yhat_xgb = model_xgb.predict(dat_test[:,:-1])
mae_xgb = mean_absolute_error(dat_test[:,-1], yhat_xgb)
print("MAE Polynomial Model = ${:,.2f}".format(1000*mae_xgb))
~~~
![image](https://user-images.githubusercontent.com/67920563/111893369-a474ee80-89d8-11eb-8442-dfd855b22dee.png)
~~~

%%timeit -n 1

mae_xgb = []

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0:-1]
  y_train = dat[idxtrain,-1]
  X_test  = dat[idxtest,0:-1]
  y_test = dat[idxtest,-1]
  model_xgb.fit(X_train,y_train)
  yhat_xgb = model_xgb.predict(X_test)
  mae_xgb.append(mean_absolute_error(y_test, yhat_xgb))
print("Validated MAE XGBoost Regression = ${:,.2f}".format(1000*np.mean(mae_xgb)))
~~~
![image](https://user-images.githubusercontent.com/67920563/111893376-b5bdfb00-89d8-11eb-98c7-4d35d8c31628.png)

## Testing Alpha Values
~~~
#alpha
a_range= np.linspace(0.01, 100)
test_mae=[]
for a in a_range:
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=1,alpha=a,gamma=1,max_depth=10)
  model_xgb.fit(dat_train[:,:-1],dat_train[:,-1])
  yhat_xgb = model_xgb.predict(dat_test[:,:-1])
  mae_xgb = mean_absolute_error(dat_test[:,-1], yhat_xgb)
  test_mae.append(mae_xgb)
~~~
~~~
 import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
ax.scatter(a_range, test_mae)
ax.plot(a_range, test_mae, c='red')
~~~

# Kernel regression from StatsModels

~~~
# This is important: update the statsmodels package
! pip install --upgrade Cython
! pip install --upgrade git+https://github.com/statsmodels/statsmodels
import statsmodels.api as sm

from statsmodels.nonparametric.kernel_regression import KernelReg
model_KernReg = KernelReg(endog=dat_train[:,-1],exog=dat_train[:,:-1],var_type='ccccccccccc')

yhat_sm_test, y_std = model_KernReg.fit(dat_test[:,:-1])

mae_sm = mean_absolute_error(dat_test[:,-1], yhat_sm_test)
print("MAE StatsModels Kernel Regression = ${:,.2f}".format(1000*mae_sm))
~~~
![image](https://user-images.githubusercontent.com/67920563/111895244-ab572d80-89e7-11eb-87e7-07e03c5527f4.png)

# Random Forest Regressor
~~~
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1234, max_depth=10,n_estimators=100)

model.fit(X,y)

importances = model.feature_importances_
indices = np.argsort(importances)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
~~~
![image](https://user-images.githubusercontent.com/67920563/111895278-e9545180-89e7-11eb-94b8-a7e309a95a3c.png)
~~~
from sklearn import model_selection
from sklearn import metrics

mae_rf=[]
cv = model_selection.KFold(n_splits=3)

for idxtrain, idxtest in kf.split(dat):
  X_train = dat[idxtrain,0:-1]
  y_train = dat[idxtrain,-1]
  X_test  = dat[idxtest,0:-1]
  y_test = dat[idxtest,-1]

  # For training, fit() is used
  model.fit(X_train, y_train)

  # Default metric is R2 for regression, which can be accessed by score()
  model.score(X_test, y_test)

  # For other metrics, we need the predictions of the model
  y_pred = model.predict(X_test)
  metrics.mean_squared_error(y_test, y_pred)
  metrics.r2_score(y_test, y_pred)
 
  mae_rf.append(mean_absolute_error(y_test, y_pred))
  print("Validated MAE Random Forest Regression = ${:,.2f}".format(1000*np.mean(mae_rf)))
  ~~~
  ![image](https://user-images.githubusercontent.com/67920563/111895373-87481c00-89e8-11eb-8d88-c09f2009b380.png)


