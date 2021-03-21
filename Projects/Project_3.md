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

# Ridge Regression
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


