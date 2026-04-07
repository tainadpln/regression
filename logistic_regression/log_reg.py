import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.random.randn(num_features)
        self.b = np.random.randn()

        for _ in range(self.n_iters):
            z = np.dot(X, self.w) + self.b
            y_predicted = self.sigmoid(z)

            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iters}, Loss: {loss:.4f}")

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y = self.sigmoid(z)
        y_pred = [1 if i > 0.5 else 0 for i in y]
        return np.array(y_pred)