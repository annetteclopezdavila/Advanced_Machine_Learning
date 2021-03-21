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


