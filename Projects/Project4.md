# Introducing the Data

~~~
#Mount google drive to access data  
from google.colab import drive
drive.mount('/content/drive')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 120

!pip install pygam

# general imports
import numpy as np
import pandas as pd
from pygam import LinearGAM
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import pyplot

df = pd.read_csv('/content/drive/MyDrive/CASP(1).csv')
df
~~~

![image](https://user-images.githubusercontent.com/67920563/114226574-b5fb4780-9941-11eb-8919-ed6df2a8675a.png)

~~~
features = ['F1', 'F2','F3','F4','F5', 'F6','F7','F8','F9']
X = np.array(df[features])
y = np.array(df['RMSD']).reshape(-1,1)
Xdf = df[features]
Xdf.shape

X_train, X_test, y_train, y_test = tts(X,y,test_size=0.1,random_state=2021)
scale = StandardScaler()

Xs_train = scale.fit_transform(X_train)
Xs_test =scale.transform(X_test)
~~~

# Generalized Additive Modeling (GAM)
6 splines
~~~
#fit the GAM with 6 splines
gam = LinearGAM(n_splines=6).gridsearch(Xs_train, y_train,objective='GCV')
gam.summary()
~~~
![image](https://user-images.githubusercontent.com/67920563/114226726-f3f86b80-9941-11eb-8b12-657cbf421dd6.png)
### RMSE
~~~
from sklearn.metrics import mean_squared_error
yhat=gam.predict(Xs_test)
rms = mean_squared_error(y_test, yhat, squared=False)
rms
~~~
![image](https://user-images.githubusercontent.com/67920563/114226775-0a9ec280-9942-11eb-88b9-1f0986d7b0f4.png)
### R2
~~~
from sklearn.metrics import r2_score as R2
yhat=gam.predict(Xs_test)
R_2= R2(y_test, yhat)
R_2
~~~
![image](https://user-images.githubusercontent.com/67920563/114230190-d11c8600-9946-11eb-836d-8aff769dfa45.png)



~~~
plt.rcParams['figure.figsize'] = (28, 8)
fig = plt.figure()
titles = df[features].columns

fig.set_figheight(16)
fig.set_figwidth(8)

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    ax = fig.add_subplot(6, 2, i+1)
    ax.plot(XX[:, term.feature], pdep)
    ax.plot(XX[:, term.feature], confi, c='r', ls='--')
    ax.set_title(titles[i])
    fig.tight_layout()
plt.show()
~~~
![image](https://user-images.githubusercontent.com/67920563/114226809-15595780-9942-11eb-94ad-e833c20ee122.png)

## Different Splines
~~~
listt=[]
listr=[]
for i in range(4, 30):
  gam = LinearGAM(n_splines=i).gridsearch(Xs_train, y_train,objective='GCV')

  yhat=gam.predict(Xs_test)
  rms = mean_squared_error(y_test, yhat, squared=False)
  R_2= R2(y_test, yhat)
  listt.append(rms)
  listr.append(R_2)
~~~


### RMSE Plot Splines
~~~
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
a_range=range(4, 30)
ax.scatter(a_range, listt)
ax.plot(a_range, listt, c='red')  
min(listt)
~~~
![image](https://user-images.githubusercontent.com/67920563/114231778-e1356500-9948-11eb-8632-3d0d6f9db8ad.png)
![image](https://user-images.githubusercontent.com/67920563/114232006-36717680-9949-11eb-8fd7-c5d4e32f40f7.png)



### R2 Plot Splines
~~~
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(8,6))
a_range=range(4, 30)
ax.scatter(a_range, listt)
ax.plot(a_range, listt, c='red')
~~~
![image](https://user-images.githubusercontent.com/67920563/114231920-1b066b80-9949-11eb-8e94-4f50fd588ee4.png)



## KFold Split
~~~
def do_kfold(X,y,k,rs, n_splines):
  PE_internal_validation=[]
  PE_external_validation=[]
  kf=KFold(n_splits=k, shuffle=True, random_state=rs)
  for idxtrain, idxtest in kf.split(X):
    X_train=X[idxtrain,:]
    y_train=y[idxtrain]
    X_test=X[idxtest,:]
    y_test=y[idxtest]
    gam = LinearGAM(n_splines=n_splines).gridsearch(X_train, y_train,objective='GCV')
    yhat_test=gam.predict(X_test)
    yhat_train=gam.predict(X_train)
    PE_internal_validation.append(MAE(y_train,yhat_train))
    PE_external_validation.append(MAE(y_test,yhat_test))
  return np.mean(PE_internal_validation), np.mean(PE_external_validation)
  
do_kfold(X,y,10,2021,6)  
~~~
![image](https://user-images.githubusercontent.com/67920563/114232364-c0214400-9949-11eb-9a80-b706ef75dcf9.png)


  
  












