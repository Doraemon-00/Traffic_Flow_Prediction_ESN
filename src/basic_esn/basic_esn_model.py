import numpy as np

class BasicESN:
    def __init__(self, input_size, reservoir_size, output_size, 
                 target_spectral_radius=0.95, leaking_rate=1.0, 
                 sparsity=0.1, input_scaling=0.1, activation='tanh'):
        """
        Echo State Network (ESN) with tunable spectral radius, leaking rate, 
        and sparse connectivity.

        Parameters:
        - input_size: Number of input neurons
        - reservoir_size: Number of reservoir neurons
        - output_size: Number of output neurons
        - target_spectral_radius: Scaling for the reservoir matrix W
        - leaking_rate: Controls how fast the reservoir state updates
        - sparsity: Ratio of zero values in W to create a sparse matrix
        - input_scaling: Scales the input weights W_in
        - activation: Activation function ('tanh' or 'relu')
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = target_spectral_radius
        self.leaking_rate = leaking_rate
        self.activation = activation

        # Initialize input weights
        self.Win = np.random.uniform(-1, 1, (reservoir_size, input_size)) * input_scaling  

        # Create sparse reservoir weight matrix
        self.W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        mask = np.random.rand(reservoir_size, reservoir_size) > sparsity  # Apply sparsity mask
        self.W[mask] = 0  

        # Normalize spectral radius
        spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / spectral_radius  

        # Initialize readout weights
        self.Wout = np.zeros((output_size, reservoir_size))

        # Initialize reservoir state randomly
        self.x = np.random.rand(self.reservoir_size) - 0.5  

    def update(self, u):
        """
        Update the reservoir state given input u.
        """
        pre_activation = np.dot(self.Win, u) + np.dot(self.W, self.x)

        if self.activation == 'tanh':
            pre_activation = np.tanh(pre_activation)
        elif self.activation == 'relu':
            pre_activation = np.maximum(0, pre_activation)

        self.x = (1 - self.leaking_rate) * self.x + self.leaking_rate * pre_activation
