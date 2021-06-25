---
layout: page
title: Summary & Notes
cover-img: /assets/img/NNN.PNG
---
## Machine Learning

- What is cross-validation? How is it applied?
  - A method for measuring the accuracy of a model
  - When we create a model with a particular dataset, we cannot assume that the data represents all the possible scenarios. Thus, we must test the model on data that it has never seen before. We can do this by splitting the dataset into training and testing sets; we will train the model on the training set and then evaluate it with the testing set. 
  - There are several types of cross-validation methods such as k-fold cross validation, stratified k-fold, leave-p-out cross validation, and the holdout method.
  
- Is it better to design robust or accurate algorithms?
  -   It depends on the model but there are tradeoffs:
    - A model that is too simple may not capture the complexities of the system (underfitting)
    - A model that is too complex may capture patterns that are more specialized to the dataset and thus not general enough to accurately work on other datasets (overfitting)
How to define/select metrics?
Explain what regularization is and why it is useful. What are the benefits and drawbacks of specific methods, such as ridge regression and lasso?
Explain what a local optimum is and why it is important in a specific context, such as K-means clustering. What are specific ways of determining if you have a local optimum problem? What can be done to avoid local optima?
Assume you need to generate a predictive model using multiple regression. Explain how you intend to validate this model
Explain what precision and recall are. How do they relate to the ROC curve?
What is latent semantic indexing? What is it used for? What are the specific limitations of the method?
Explain what resampling methods are and why they are useful
What is principal component analysis? Explain the sort of problems you would use PCA for. Also explain its limitations as a method
Explain what a false positive and a false negative are. Why is it important these from each other? Provide examples when false positives are more important than false negatives, false negatives are more important than false positives and when these two types of errors are equally important
What is the difference between supervised learning and unsupervised learning? Give concrete examples
What does NLP stand for?
What are feature vectors?
When would you use random forests Vs SVM and why?
How do you take millions of users with 100’s transactions each, amongst 10k’s of products and group the users together in meaningful segments?
How do you know if one algorithm is better than other?
How do you test whether a new credit risk scoring model works?
What is: collaborative filtering, n-grams, cosine distance?
What is better: good data or good models? And how do you define “good”? Is there a universal good model? Are there any models that are definitely not so good?
Why is naive Bayes so bad? How would you improve a spam detection algorithm that uses naive Bayes?
What are the drawbacks of linear model? Are you familiar with alternatives (Lasso, ridge regression, boosted trees)?
Do you think 50 small decision trees are better than a large one? Why?
Why is mean square error a bad measure of model performance? What would you suggest instead?
How can you prove that one improvement you’ve brought to an algorithm is really an improvement over not doing anything? Are you familiar with A/B testing?
What do you think about the idea of injecting noise in your data set to test the sensitivity of your models?
Do you know / used data reduction techniques other than PCA? What do you think of step-wise regression? What kind of step-wise techniques are you familiar with?
How would you define and measure the predictive power of a metric?
Do we always need the intercept term in a regression model?
What are the assumptions required for linear regression? What if some of these assumptions are violated?
What is collinearity and what to do with it? How to remove multicollinearity?
How to check if the regression model fits the data well?
What is a decision tree?
What impurity measures do you know?
What is random forest? Why is it good?
How do we train a logistic regression model? How do we interpret its coefficients?
What is the maximal margin classifier? How this margin can be achieved and why is it beneficial? How do we train SVM?
What is a kernel? Explain the kernel trick
Which kernels do you know? How to choose a kernel?
Is it beneficial to perform dimensionality reduction before fitting an SVM? Why or why not?
(What is an Artificial Neural Network?) What is back propagation?
What is curse of dimensionality? How does it affect distance and similarity measures?
What is Ax=b? How to solve it?
How do we multiply matrices?
What is singular value decomposition? What is an eigenvalue? And what is an eigenvector?
What’s the relationship between PCA and SVD?
Can you derive the ordinary least square regression formula?
What is the difference between a convex function and non-convex?
What is gradient descent method? Will gradient descent methods always converge to the same point?
What the Newton’s method is?
Imagine you have N pieces of rope in a bucket. You reach in and grab one end-piece, then reach in and grab another end-piece, and tie those two together. What is the expected value of the number of loops in the bucket?

