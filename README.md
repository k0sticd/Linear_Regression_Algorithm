# Linear_Regression_Algorithm
Implementation of the basic Linear Regression algorithm

Implementation of the basic Linear Regression algorithm, with prior attribute normalization, and error optimization using gradient descent. The code is organized through class methods.

Added functionality: To prevent overfitting, the regression parameters was being regularized by adding a penalty to the error function being optimized, and by computing the gradient of the new objective function (error + penalty). The regularization penalty, which is added to the squared error function, is Ridge (L2 regularization). λ represents the regularization parameter, which the user can adjust.

Added functionality: Option to train the model using online learning, i.e., using stochastic gradient descent, which is computed based on a single (random) instance

Added functionality: Ability to automatically determine the parameter 𝛼 (learning step) to control the learning rate of gradient descent
