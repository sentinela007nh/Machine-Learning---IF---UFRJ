import numpy


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []
        self.b = []
        self.L = len(layers)

        for i in range(self.L - 1):
            weight_matrix = numpy.random.randn(layers[i], layers[i + 1]) * 0.01
            bias_vector = numpy.zeros((1, layers[i + 1]))
            self.W.append(weight_matrix)
            self.b.append(bias_vector)

    def __repr__(self):
        return f"NeuralNetwork(layers={self.layers}, alpha={self.alpha})"

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            A = [X]
            for i in range(self.L - 1):
                Z = numpy.dot(A[i], self.W[i]) + self.b[i]
                A.append(self.sigmoid(Z))

            error = y - A[-1]
            D = [error * self.sigmoid_derivative(A[-1])]

            for i in range(self.L - 2, 0, -1):
                delta = D[0].dot(self.W[i].T) * self.sigmoid_derivative(A[i])
                D.insert(0, delta)

            for i in range(self.L - 1):
                self.W[i] += A[i].T.dot(D[i]) * self.alpha
                self.b[i] += numpy.sum(D[i], axis=0, keepdims=True) * self.alpha

            if epoch % 100 == 0:
                loss = numpy.mean(numpy.square(error))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        A = X
        for i in range(self.L - 1):
            Z = numpy.dot(A, self.W[i]) + self.b[i]
            A = self.sigmoid(Z)
        return A

    def calculate_loss(self, X, y):
        A = self.predict(X)
        loss = numpy.mean(numpy.square(y - A))
        return loss
