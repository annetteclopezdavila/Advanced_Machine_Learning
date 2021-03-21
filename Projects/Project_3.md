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

# Nonlinear Ridge Regression
## Testing Different K -Fold Values
~~~
model = Ridge(alpha=20)
DoKFold_SK(X,y,model,10)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891805-05e29080-89cc-11eb-86fb-116eb4ae0f56.png)
~~~
model = Ridge(alpha=20)
DoKFold_SK(X,y,model,30)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891812-13981600-89cc-11eb-811f-039ded3fc42a.png)
~~~
model = Ridge(alpha=20)
DoKFold_SK(X,y,model,300)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891820-23175f00-89cc-11eb-94fd-9cb25b95817d.png)
~~~
model = Ridge(alpha=20)
DoKFold_SK(X,y,model,506)
~~~
![image](https://user-images.githubusercontent.com/67920563/111891827-2f032100-89cc-11eb-9c6c-5638fd154a01.png)
## Testing Different Alphas
~~~
#test alphas
a_range= np.linspace(0.01, 100)
test_mae=[]
for a in a_range:
  test_mae.append(DoKFold_SK(X,y,model,10))
min(test_mae)  
~~~  
![image](https://user-images.githubusercontent.com/67920563/111891865-843f3280-89cc-11eb-94ca-52aeea37c5ea.png)
~~~
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
ax.scatter(a_range, test_mae)
ax.plot(a_range, test_mae, c='red')
~~~
![image](https://user-images.githubusercontent.com/67920563/111891885-b5b7fe00-89cc-11eb-9fed-0de53f15db28.png)

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

 





