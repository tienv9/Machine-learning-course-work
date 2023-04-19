########## >>>>>> Eric Vo - tvo12. 

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent




import numpy as np
import math
import sys
sys.path.append("..")

from code_misc.utils import MyUtils



class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1

        

    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''

        # remove the pass statement and fill in the code. 
        
        self.degree = degree
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis = 1)
        
        if SGD:
            self._SGD(X, y, lam, eta, iterations, mini_batch_size)
        else:
            self._BGD(X, y, lam, eta, iterations)
        
        
        
    def _BGD(self, X, y, lam, eta, iterations):
        n, d = np.shape(X)
        w = np.zeros((d, 1))
        XT = X.T
        
        r = 1 - (2 * lam * (eta / n))
        
        for i in range(iterations):
            s = y * (X @ w)
            v = LogisticRegression._v_sigmoid(-s)
            w = r * w + ((eta / n) * (XT @ (y * v)))
            
        self.w = w
        
    def _SGD(self, X, y, lam, eta, iterations, mini_batch_size):
        n, d = np.shape(X)
        m = mini_batch_size
        w = np.zeros((d,1))
        
        if m > n or m < 1:
            m = n
        
        
        
        
        roll = False
        
        if roll:
            rol = np.append(X, y, axis = 1)
            np.random.shuffle(rol)
            X = rol[:, :d]
            y = rol[:, d:]
        
        
        
        
        a = 0
        b = a + m
        c = a
        
        r = 1 - (2 * lam * (eta / m))
        
        for i in range(iterations):
            ym = y[c:b, :]
            xm = X[c:b, :]
            
            s = ym * (xm @ w)
            
            v = LogisticRegression._v_sigmoid(-s)
            
            w = ((eta / m) * ((ym * v).T @ xm)).T + (r * w)
            
            if n != m: 
                a += m
                b = a + m
                c = a 
                
                if b >= n and a < n:
                    c = a
                    a = -m
                    b = n
                if b > n:
                    a = 0
                    b = a + m
        self.w = w 
        
        
        
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
    
        
        # remove the pass statement and fill in the code. 
        
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis = 1)
    
        return LogisticRegression._v_sigmoid(X @ self.w)
    
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''

        # remove the pass statement and fill in the code.         
        count = 0 
        error = self.predict(X)
        
        for a, b in zip(error, y): 
            if a > 0.5:
                c = 1
            else:
                c = -1
            
            if c != b:
                count += 1
    
        return count

    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
            
        # Hint: use the np.vectorize API

        # remove the pass statement and fill in the code.         
        v = np.vectorize(LogisticRegression._sigmoid)
        return v(s)
        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''

        # remove the pass statement and fill in the code.         
        e = np.exp(-s)
        return (1 / (1 + e))
