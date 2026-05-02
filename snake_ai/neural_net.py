import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes=[11, 16, 3]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = self.relu(x)
        return x

    def predict(self, state):
        x = np.array([state], dtype=float)
        out = self.forward(x)
        return int(np.argmax(out))

    def get_weights_flat(self):
        parts = [w.flatten() for w in self.weights] + \
                [b.flatten() for b in self.biases]
        return np.concatenate(parts)

    def set_weights_flat(self, flat):
        idx = 0
        for i, w in enumerate(self.weights):
            size = w.size
            self.weights[i] = flat[idx:idx+size].reshape(w.shape)
            idx += size
        for i, b in enumerate(self.biases):
            size = b.size
            self.biases[i] = flat[idx:idx+size].reshape(b.shape)
            idx += size

    def copy(self):
        import copy
        nn = NeuralNetwork(self.layer_sizes)
        nn.weights = copy.deepcopy(self.weights)
        nn.biases = copy.deepcopy(self.biases)
        return nn