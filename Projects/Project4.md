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
~~~
from sklearn.metrics import mean_squared_error
yhat=gam.predict(Xs_test)
rms = mean_squared_error(y_test, yhat, squared=False)
rms
~~~
![image](https://user-images.githubusercontent.com/67920563/114226775-0a9ec280-9942-11eb-88b9-1f0986d7b0f4.png)

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














