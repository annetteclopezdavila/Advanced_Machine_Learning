---
layout: page
title: Project 4
subtitle: Real Data Applications of GAM and Nadaraya-Watson Regression
---
This project will be showing the application of Generalized Additive Models and Nadaraya-Watson Regression.

# Introducing the Data
This dataset examines physicochemical properties of protein tertiary structure. It is a multivariate dataset with 45,730 samples. Our Attributes are as follows:

### *Target*:
RMSD-Size of the residue.

### *Features*:
- F1 - Total surface area.
- F2 - Non polar exposed area.
- F3 - Fractional area of exposed non polar residue.
- F4 - Fractional area of exposed non polar part of residue.
- F5 - Molecular mass weighted exposed area.
- F6 - Average deviation from standard exposed area of residue.
- F7 - Euclidian distance.
- F8 - Secondary structure penalty.
- F9 - Spacial Distribution constraints (N,K Value).

## Setting up the DataFrame
Below is the python code used to set up the dataframe:
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

We then can divide the dataset into its independent and dependent variables, as well as splitting the training and testing sets and standardizing the data.
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

![image](https://user-images.githubusercontent.com/67920563/114255223-01cde100-9982-11eb-8355-8413c9598e56.png)

Additive models estimate an additive approximation to multivariate regression functions. They can deal with highly non-linear and non-monotonic relationships by using the data to shape the predictor functions. This type of model helps avoid the curse of dimensionality by using univariate smoothers. The individual term estimates will also explain the relationship between variables. GAM models should be used when non-linearity in partial residual plots suggest the need for semi-parametric models(relationships among variables are not restricted to any shape generally).

![image](https://user-images.githubusercontent.com/67920563/114255092-31c8b480-9981-11eb-887a-6fb1671dfecf.png)

GAM models will separate predictors into sections and will fit the data in each section using spline functions. All the functions are then added to predict the link function. The link function is then smoothed by LOESS.

## Python Code
Let us attempt to fit a GAM with the pyGAM library. Generally, high splines are used (~20), but let us begin with 6.
~~~
#fit the GAM with 6 splines
gam = LinearGAM(n_splines=6).gridsearch(Xs_train, y_train,objective='GCV')
gam.summary()
~~~
Below we see the summary of our model: we can use AIC, GCV, and R-Squared indeces to help us understand our model. This also tells us the smoothing pentalty used which is lambda=0.001. This controls the strength of the regularization penalty on each term. 
![image](https://user-images.githubusercontent.com/67920563/114226726-f3f86b80-9941-11eb-8b12-657cbf421dd6.png)
### RMSE
For this particular project, we will be comparing our models with the root mean square error. The RMSE represents the standard deviation of the prediction errors, known as residuals. The residuals show how far a data point is from the regression line. The RMSE formula is as follows:

![image](https://user-images.githubusercontent.com/67920563/114255926-8faacb80-9984-11eb-9a6f-500244f0f775.png)

In order to find the RMSE, we can use the sklearn library and add the parameter "squared=False" in order to find the RMSE instead of the MSE. This is shown below:
~~~
from sklearn.metrics import mean_squared_error
yhat=gam.predict(Xs_test)
rms = mean_squared_error(y_test, yhat, squared=False)
rms
~~~
We must try to minimize our RMSE value. Our RMSE must be evaluated in context of the range of the dataset. Our RMSE for this model at 6 splines is:

![image](https://user-images.githubusercontent.com/67920563/114226775-0a9ec280-9942-11eb-88b9-1f0986d7b0f4.png)
### R^2
We can also use R^2 to evaluate our models. R-Squared values show how well the data fits the regression model by determining the proportion of variance in y that is captured by the model. 
~~~
from sklearn.metrics import r2_score as R2
yhat=gam.predict(Xs_test)
R_2= R2(y_test, yhat)
R_2
~~~
The closer an R-squared value is to one or negative one, the better the regression describes the data. Our R-squared value here is:
![image](https://user-images.githubusercontent.com/67920563/114230190-d11c8600-9946-11eb-836d-8aff769dfa45.png)


## Residual Plots
Residual values measure the difference between the regression lines and the data points. A residual plot will have the residuals on the y axis and the y target on the x axis. Let us now see the residual plots for this model:
~~~
residuals=gam.deviance_residuals(X_train,y_train)
residualst=gam.deviance_residuals(X_test,y_test)

import matplotlib.pyplot as plt
plt.scatter(residuals, y_train, c="pink", s=20, alpha=0.5)
plt.scatter(residualst, y_test, c="green", s=20, alpha=0.5)

plt.xlabel("Predicted Value")
plt.ylabel("Deviance Residuals")
plt.show()
~~~
We can also plot a histogram of our residuals in order to see whether the variance has a particular distribution. A histogram will clarify the frequency of the residuals.
~~~
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.histograms import histogram

plt.hist(residuals, label='residuals train', alpha=.5, color='pink')
plt.hist(residualst, label='residuals test', alpha=.5, color='green')
plt.legend()
plt.show()

~~~
From the Residuals Plot, we can see that there is a large cluster that travels diagonally. This is a warning sign that our model may not be a good fit for this dataset. A good fitting model will have a residual plot with scattered points. We can note that the pink dots are the training set and the green are the testing set.

![image](https://user-images.githubusercontent.com/67920563/114253347-d940e980-9977-11eb-8c8e-703154361eb9.png)

Our histogram below is skewed to the right:
![image](https://user-images.githubusercontent.com/67920563/114253388-fc6b9900-9977-11eb-8c04-a911162321fa.png)

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

# Nadaraya-Watson Regression
~~~
"""
Nadaraya-Watson Regression.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

# TODO : evaluate overwrite of K (kernel)
from __future__ import division

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.scorer import check_scoring
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import GridSearchCV, ParameterGrid, check_cv
from sklearn.model_selection._search import _check_param_grid
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils import check_X_y

def squared_norm(arr):
    """Compute the square frobenius/vector norm.

    Parameters
    ----------
    arr : np.ndarray
        Array of which we compute the norm.

    Returns
    -------
    norm: float
    """
    arr = arr.ravel(order='K')
    return np.dot(arr, arr)

class NadarayaWatson(BaseEstimator, RegressorMixin):
    """NadarayaWatson Estimator.

    Parameters
    ----------
    kernel : string or callable, default="linear"
        Kernel mapping used to compute weights.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Ignored by other kernels.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Additional parameters for kernel function passed as callable object.

    Notes
    -----
    See `sklearn.kernel_ridge <http://scikit-learn.org/stable/modules/
    generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.
    KernelRidge>`_, for more info: Kernel Ridge Regression estimator from
    which the structure of this estimator is based.

    Examples
    --------
    >>> import numpy as np
    >>> from nadaraya_watson import NadarayaWatson
    >>> # generate some fake data
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> # fit regressor
    >>> reg = NadarayaWatson()
    >>> reg.fit(X, y)
    NadarayaWatson(coef0=1, degree=3, gamma=None, kernel='linear',
            kernel_params=None)
    """

    def __init__(self, kernel="linear", degree=3,
                 coef0=1, gamma=None, kernel_params=None):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, y=None):
        """Gets kernel matrix."""
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}

        return pairwise_kernels(X, y, metric=self.kernel,
                                filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _check_fit_arrays(self, X, y, sample_weight=None):
        """Checks fit arrays and scales y if sample_weight is not None."""
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"),
                         multi_output=True, y_numeric=True)

        if sample_weight is not None and not isinstance(sample_weight, float):
            # TODO: break up?
            sample_weight = check_array(sample_weight, ensure_2d=False)

            # do not want to rescale X!!!!
            y = np.multiply(sample_weight[:, np.newaxis], y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        return X, y

    def fit(self, X, y, sample_weight=None):
        """Fit Nadaraya Watson estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples, n_features)
            Target values.

        Returns
        -------
        self : returns an instance of self
        """
        X, y = self._check_fit_arrays(X, y, sample_weight)

        self.X_ = X
        self.y_ = y

        return self

    @staticmethod
    def _normalize_kernel(K, overwrite=False):
        """Normalizes kernel to have row sum == 1 if sum != 0"""
        factor = K.sum(axis=1)

        # if kernel has finite support, do not divide by zero
        factor[factor == 0] = 1

        # divide in place
        if overwrite:
            return np.divide(K, factor[:, np.newaxis], K)

        return K/factor[:, np.newaxis]


    def get_weights(self, X):
        """Return model weights."""
        check_is_fitted(self, ["X_", "y_"])
        K = self._get_kernel(X, self.X_)

        return self._normalize_kernel(K, overwrite=True)

    def predict(self, X):
        """Predict using the Nadaraya Watson model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        C : array, shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self, ["X_", "y_"])

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        w = self.get_weights(X)

        # TODO: evaluate sklearn.utils.extmath.safe_sparse_dot()
        if issparse(self.y_):
            # has to be of form sparse.dot(dense)
            # more efficient than w.dot( y_.toarray() )
            return self.y_.T.dot(w.T).T

        return w.dot(self.y_)

    @property
    def nodes(self):
        """Nodes (data)"""
        check_is_fitted(self, ["X_", "y_"])
        return self.y_


class _NadarayaWatsonLOOCV(NadarayaWatson):
    """Nadaraya watson with built-in Cross-Validation

    It allows efficient Leave-One-Out cross validation

    This class is not intended to be used directly. Use NadarayaWatsonCV instead.
    """
    def __init__(self, param_grid, scoring=None, store_cv_scores=False):
        #TODO: check _check_param_grid in proper spot
        self.param_grid = param_grid
        self.scoring = scoring
        self.store_cv_scores = store_cv_scores
        _check_param_grid(param_grid)

    @property
    def _param_iterator(self):
        return ParameterGrid(self.param_grid)

    def _errors_and_values_helper(self, K):
        """Helper function to avoid duplication between self._errors and
        self._values.

        fill diagonal with 0, renormalize
        """
        np.fill_diagonal(K, 0)
        S = self._normalize_kernel(K, overwrite=True)

        return S

    def _errors(self, K, y):
        """ mean((y - Sy)**2) = mean( ((I-S)y)**2 )"""
        S = self._errors_and_values_helper(K)

        # I - S (S has 0 on diagonal)
        S *= -1
        np.fill_diagonal(S, 1.0)

        mse = lambda x: squared_norm(x) / x.size
        return mse(S.dot(y))

    def _values(self, K, y):
        """ prediction """
        S = self._errors_and_values_helper(K)

        return S.dot(y)

    def fit(self, X, y, sample_weight=None):
        """Fit the model using efficient leave-one-out cross validation"""
        X, y = self._check_fit_arrays(X, y, sample_weight)

        candidate_params = list(self._param_iterator)

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        # error = scorer is None
        error = self.scoring is None

        if not error:
            # scorer wants an object to make predictions
            # but are already computed efficiently by _NadarayaWatsonCV.
            # This identity_estimator will just return them
            def identity_estimator():
                pass
            identity_estimator.predict = lambda y_pred: y_pred

        cv_scores = []
        for candidate in candidate_params:
            # NOTE: a bit hacky, find better way
            K = NadarayaWatson(**candidate)._get_kernel(X)
            if error:
                # NOTE: score not error!
                score = -self._errors(K, y)
            else:
                y_pred = self._values(K, y)
                score = scorer(identity_estimator, y, y_pred)
            cv_scores.append(score)

        self.n_splits_ = X.shape[0]
        self.best_index_ = np.argmax(cv_scores)
        self.best_score_ = cv_scores[self.best_index_]
        self.best_params_ = candidate_params[self.best_index_]
        if self.store_cv_scores:
            self.cv_scores_ = cv_scores

        return self


class NadarayaWatsonCV(NadarayaWatson):
    """NadarayaWatson Estimator with built in Leave-one-out cross validation.

    By default, it performs Leave-one-out cross validation efficiently, but
    can accept cv argument to perform arbitrary cross validation splits.

    Parameters
    ----------
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter settings.

    scoring : string, callable or None, optional, default: None
        A string (see sklearn.model_evaluation documentation) or a scorer
        callable object / function with signature
        ``scorer(estimator, X, y)``

    cv : int, cross-validation generator or an iterable, optional, default: None
        Determines the cross-validation splitting strategy. If None, perform
        efficient leave-one-out cross validation, else use
        sklearn.model_selection.GridSearchCV.

    store_cv_scores : boolean, optional, default=False
        Flag indicating if the cross-validation values should be stored in
        `cv_scores_` attribute. This flag is only compatible with `cv=None`.

    Attributes
    ----------
    cv_scores_ : array, shape = (n_samples, ~len(param_grid))
        Cross-validation scores for each candidate parameter (if
        `store_cv_scores=True` and `cv=None`)

    best_score_ : float
        Mean cross-validated score of the best performing estimator.

    n_splits_ : int
        Number of cross-validation splits (folds/iterations)

    Examples
    --------
    >>> import numpy as np
    >>> from nadaraya_watson import NadarayaWatson
    >>> # generate some fake data
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> # fit regressor
    >>> param_grid = [dict(kernel=['linear'], degree=np.arange(1, 4)),
    ...               dict(kernel=['rbf'], gamma=np.logspace(-1, 1, 3))]
    >>> reg = NadarayaWatsonCV(param_grid)
    >>> reg.fit(X, y)
    NadarayaWatsonCV(coef0=1, cv=None, degree=3, gamma=1.0, kernel='rbf',
             kernel_params=None,
             param_grid=[{'kernel': ['linear'], 'degree': array([1, 2, 3])},
                          {'kernel': ['rbf'], 'gamma': array([ 0.1,  1. , 10. ])}],
             scoring=None, store_cv_scores=False)
    """

    def __init__(self, param_grid, scoring=None, cv=None, store_cv_scores=False,
                 kernel="linear", degree=3, coef0=1, gamma=None, kernel_params=None):
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.store_cv_scores = store_cv_scores

        # NadarayaWatson kwargs :: for compatibility
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _update_params(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)

    def fit(self, X, y, sample_weight=None):
        """Fit Nadaraya Watson estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples, n_features)
            Target values.

        Returns
        -------
        self : returns an instance of self
        """
        if self.cv is None:
            estimator = _NadarayaWatsonLOOCV(param_grid=self.param_grid,
                                             scoring=self.scoring,
                                             store_cv_scores=self.store_cv_scores)
            estimator.fit(X, y, sample_weight=sample_weight)
            self.best_score_ = estimator.best_score_
            self.n_splits_ = estimator.n_splits_
            best_params_ = estimator.best_params_
            if self.store_cv_scores:
                self.best_index_ = estimator.best_index_
                self.cv_scores_ = estimator.cv_scores_
        else:
            if self.store_cv_scores:
                raise ValueError("cv!=None and store_cv_score=True "
                                 "are incompatible")
            gs = GridSearchCV(NadarayaWatson(), self.param_grid,
                              cv=self.cv, scoring=self.scoring, refit=True)
            gs.fit(X, y, sample_weight=sample_weight)
            estimator = gs.best_estimator_
            self.n_splits_ = gs.n_splits_
            self.best_score_ = gs.best_score_
            best_params_ = gs.best_params_

        # set params for predict
        self._update_params(best_params_)

        # store data for predict
        self.X_ = X
        self.y_ = y

        return self
        
~~~

~~~
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

%matplotlib inline
%config InlineBackend.

param_grid=dict(kernel=["rbf"],gamma=np.linspace(30, 50, 50))
model = GridSearchCV(NadarayaWatson(), scoring='neg_mean_squared_error', cv=5, param_grid=param_grid)

model=NadarayaWatson(kernel='rbf', gamma=35)
model.fit(X_train, y_train)

yhat= model.predict(X_test)
~~~

~~~
R2(y_test, yhat)
~~~
![image](https://user-images.githubusercontent.com/67920563/114253522-93385580-9978-11eb-847e-9dd917ab714b.png)

~~~
mse(y_test, yhat)
~~~
![image](https://user-images.githubusercontent.com/67920563/114253552-a0554480-9978-11eb-82db-4149cd4c4258.png)

~~~
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(model)
model = GridSearchCV(NadarayaWatson(), scoring='neg_mean_squared_error', cv=5, param_grid=param_grid)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof() 
~~~
![image](https://user-images.githubusercontent.com/67920563/114254070-4bff9400-997b-11eb-82e4-33ad5a71a989.png)

~~~
def DoKFold(X,y,k):
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=2021)
  for idxtrain, idxtest in kf.split(X):
    param_grid=dict(kernel=["rbf"],gamma=np.linspace(30, 50, 50))
    model = GridSearchCV(NadarayaWatson(), scoring='neg_mean_squared_error', cv=5, param_grid=param_grid)
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model.fit(X_train,y_train)
    yhat_test = model.predict(X_test)
    PE.append(MAE(y_test,yhat_test))
  return np.mean(PE)
  
DoKFold(X,y,10)  
  ~~~




  












