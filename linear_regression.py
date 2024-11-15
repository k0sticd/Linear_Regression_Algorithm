import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_reg=0.1, stochastic=False, learning_rate_options=None):
        self._validate_params(learning_rate, num_iterations, lambda_reg)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.stochastic = stochastic
        self.learning_rate_options = learning_rate_options #or [0.001, 0.01, 0.1, 1.0]
        self.beta = None
        self.mean = None
        self.std = None
    
    def _validate_params(self, learning_rate, num_iterations, lambda_reg):
        if not (0 < learning_rate <= 1):
            raise ValueError("Learning rate must be between 0 and 1.")
        if num_iterations <= 0:
            raise ValueError("Number of iterations must be a positive integer.")
        if lambda_reg < 0:
            raise ValueError("Regularization parameter lambda must be non-negative.")
    
    def normalize(self, X, fit=True):
        self.mean = X.mean()
        self.std = X.std()
        self.std = np.array(self.std)
        X_normalized = (X - self.mean) / self.std
        return X_normalized
        
    def learn(self, X, y):
        m, n = X.shape
        X = self.normalize(X, fit=True)
        X = np.c_[np.ones(m), X]  # Add a column of 1s for the intercept

        if self.learning_rate_options is not None:
            # If options for the learning rate are provided, we search for the best one
            self._find_best_learning_rate(X, y)
        else:
            # If they are not provided, we use the current learning rate
            self._train_model(X, y, self.learning_rate)

    def _find_best_learning_rate(self, X, y):
        best_lr = self.learning_rate
        min_cost = float('inf')
        best_beta = None  

        for lr in self.learning_rate_options:
            self.learning_rate = lr
            self.beta = np.random.randn(X.shape[1]) * 0.01  # Initializes beta

            cost, stop_early = self._train_model(X, y, lr)

            if cost < min_cost:
                min_cost = cost
                best_lr = lr
                best_beta = self.beta.copy()  # Save the best beta

        self.learning_rate = best_lr
        self.beta = best_beta  # Set the best beta
        print(f"Optimal learning rate found: {self.learning_rate}")

    def _train_model(self, X, y, lr):
        self.learning_rate = lr
        self.beta = np.random.randn(X.shape[1]) * 0.01  # Initialize beta

        stop_early = False
        for iteration in range(self.num_iterations):
            if self.stochastic:
                index = np.random.randint(X.shape[0])  # Randomly select a sample
                self._update_beta(X[index:index+1], y[index:index+1])
            else:
                self._update_beta(X, y)
                
            grad_norm = np.linalg.norm(self.beta)  # Calculate the gradient norm
            
            if grad_norm < 1:
                print(f"Gradient norm {grad_norm} is below threshold. Stopping early.")
                stop_early = True
                break
        
        cost = self._compute_cost(X, y)
        return cost, stop_early        
        

    def _update_beta(self, X, y):
        if self.beta is None:
            raise ValueError("Model parameters (beta) are not initialized.")
    
        m = X.shape[0]
        if X.shape[1] != self.beta.shape[0]:
            raise ValueError("Dimension mismatch between X and beta.")
    
        predictions = X.dot(self.beta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        gradient[1:] += (self.lambda_reg / m) * self.beta[1:]
        
        grad_norm = np.abs(gradient).sum()  # Compute the gradient norm
        
        print(f"Gradient norm: {grad_norm}")
        
        # Check for NaN and inf values in the gradient
        if np.any(np.isnan(gradient)):
            raise ValueError("Gradient contains NaN values.")
        if np.any(np.isinf(gradient)):
            raise ValueError("Gradient contains inf values.")
       
        self.beta -= self.learning_rate * gradient
       
        # Check for NaN and inf values in beta after the update
        if np.any(np.isnan(self.beta)) or np.any(np.isinf(self.beta)):
            raise ValueError("Beta contains NaN or inf values after update.")
    
    def _compute_cost(self, X, y):
        m = X.shape[0]
        predictions = X.dot(self.beta)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost += (self.lambda_reg / (2 * m)) * np.sum(self.beta[1:] ** 2)
        return cost

    def predict(self, X):
        X = self.normalize(X, fit=False)  # Normalize X using the same parameters as X during training
        X = np.c_[np.ones(X.shape[0]), X]  # Add a column of 1s for the intercept
        print(f"X shape in predict: {X.shape}")  # It should be (m_test, n+1)
        print(f"Beta shape: {self.beta.shape}")  # It should be (n+1,)
        return X.dot(self.beta)

def load_data(file_path, test_size=0.2, random_state=None):
    
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data('boston.csv')

model = LinearRegression(num_iterations=3000, lambda_reg=10, stochastic=False, learning_rate = 0.15)#,learning_rate=0.9 , learning_rate_options=[0.001, 0.01, 0.1, 1.0]
model.learn(X_train, y_train)
    

predictions = model.predict(X_test)
    
print("Predictions on Test Set:", predictions)