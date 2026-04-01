import numpy as np

class SoftmaxRegression:
    def __init__(self, n_classes, lr=0.01, n_iters=1000):
        self.n_classes = n_classes
        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None
        self.losses = []
    
    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    
    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)
    
    def compute_loss(self, y_pred, y):
        m = y.shape[0]
        correct_logprobs = -np.log(y_pred[np.arange(m), y] + 1e-8)
        return np.mean(correct_logprobs)
    
    def backward(self, X, y_pred, y_true):
        m = X.shape[0]
        
        y_one_hot = np.zeros((m, self.n_classes))
        y_one_hot[np.arange(m), y_true] = 1
        
        dz = y_pred - y_one_hot
        dW = np.dot(X.T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        
        return dW, db
    
    def fit(self, X, y):
        m, n_features = X.shape
        
        self.W = np.zeros((n_features, self.n_classes))
        self.b = np.zeros((1, self.n_classes))
        
        # Huấn luyện
        for i in range(self.n_iters):
            y_pred = self.forward(X)
            
            loss = self.compute_loss(y_pred, y)
            self.losses.append(loss)
            
            dW, db = self.backward(X, y_pred, y)
            
            self.W -= self.lr * dW
            self.b -= self.lr * db
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iters}, Loss: {loss:.4f}")
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X)