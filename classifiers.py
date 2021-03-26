import numpy as np
import cvxpy as cp
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

class LogisticRegression:
    
    def __init__(self, params):
        
        self.lr = params.lr
        self.num_iter = params.num_iter
        self.fit_intercept = params.fit_intercept
        self.verbose = params.verbose
        self.print_freq = params.print_freq
    
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss_function(self, y_pred, y_true):
        return (-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        # weights initialization
        self.parameters = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.parameters)
            y_pred = self.sigmoid(z)
            grad = np.dot(X.T, (y_pred - y)) / y.size
            self.parameters = self.parameters - self.lr * grad
            
            if self.verbose and i % self.print_freq == 0:
                sys.stdout.write(f'\rstep {i}/{self.num_iter} | loss = {self.loss_function(y_pred, y):.3f}')
                sys.stdout.flush()
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
    
        return self.sigmoid(np.dot(X, self.parameters))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


class softSVM:
    
    def __init__(self, X, y, params):
        
        m, n = X.shape
        self.C = params.C
        self.Q = np.diag(y) @ X @ X.T @ np.diag(y)
        self.p = -np.ones((m,1))
        self.G = np.block([[np.eye(m)],[-np.eye(m)]])
        self.h = np.block([[C*np.ones((m,1))],[-np.zeros((m,1))]])
        self.A = y.reshape(1,-1) * 1.
        self.b = np.zeros(1)

    def fit(self, X, y):
        
        Q = cvxopt_matrix(self.Q)
        p = cvxopt_matrix(self.p)
        G = cvxopt_matrix(self.G)
        h = cvxopt_matrix(self.h)
        A = cvxopt_matrix(self.A)
        b = cvxopt_matrix(self.b)
        dual_sol = cvxopt_solvers.qp(Q, p, G, h, A, b)
        dual_sol = np.array(dual_sol['x'])
        y = y.reshape(-1,1)
        self.primal_sol = ((y * dual_sol).T @ X).reshape(-1,1)
        
        ind = (dual_sol > 1e-4).flatten()
        sv_x = X[ind]
        sv_y = y[ind].reshape(-1,1)
        self.bias = sv_y - np.dot(sv_x, self.primal_sol)
        self.bias = np.mean(self.bias)
 
    def predict(self, X):
        preds = np.sign((X @ self.primal_sol) + self.bias).T
        return np.array([int(p==1) for p in preds.tolist()[0]])


class KernelRidgeRegression:
    
    def __init__(self, K, lmbda=1e-7):
        
        self.K = K
        self.m = K.shape[0]
        self.lmbda = lmbda
    
    def fit(self, y):
        
        I = np.eye(self.m)
        y = y.reshape(-1,1)
        self.alpha = np.linalg.solve(self.K+self.lmbda*self.m*I, y)
        
    def predict(self, K):
        
        preds = self.alpha.T @ K
        return np.sign(preds).reshape(-1,)


class KernelSVM:
    
    def __init__(self, K, lmbda=1e-7):
        
        self.K = K
        self.m = K.shape[0]
        self.alpha = cp.Variable(self.m)
        self.lmbda = lmbda
    
    def fit(self, y):
        
        objective = cp.Maximize(2*self.alpha.T @ y - cp.quad_form(self.alpha, self.K))
        constraints = [
            0 <= cp.multiply(y,self.alpha), 
            cp.multiply(y,self.alpha) <= 1/2/self.lmbda/self.m
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        self.alpha_star = self.alpha.value
        
    def predict(self, K):
        
        preds = self.alpha_star.T @ K
        return np.sign(preds)


def get_classsifier(x, y, params):

    if params.clf == 'lr':
        clf = LogisticRegression(params)
    elif params.clf == 'svm':
        clf = softSVM(X, y, params)      
    return clf


def get_kernel_classsifier(K, params):

    if params.clf == 'krr':
        clf = KernelRidgeRegression(K, lmbda=params.lmbda)
    elif params.clf == 'ksvm':
        clf = KernelSVM(K, lmbda=params.lmbda)      
    return clf