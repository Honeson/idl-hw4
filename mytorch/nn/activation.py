import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        self.input_shape = Z.shape
        Z_moved = np.moveaxis(Z, self.dim, -1)
        self.moved_shape = Z_moved.shape 
        C = Z_moved.shape[-1]
        Z_flat = Z_moved.reshape(-1, C)

        Z_stable = Z_flat - np.max(Z_flat, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        A_flat = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        A_moved = A_flat.reshape(self.moved_shape)

        self.A = np.moveaxis(A_moved, -1, self.dim)
        return self.A 
    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        
        # Let's move the dimension to the end.
        A_moved = np.moveaxis(self.A, self.dim, -1)
        dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
        
        moved_shape = A_moved.shape
        # Reshape input to 2D
        if len(shape) > 2:
            self.A = A_moved.reshape(-1, C)
            dLdA = dLdA_moved.reshape(-1, C)
        else:
   
            self.A = A_moved
            dLdA = dLdA_moved
        
        dLdZ = self.A * (dLdA - np.sum(dLdA * self.A, axis=1, keepdims=True))

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = A_moved.reshape(moved_shape)
            dLdZ = dLdZ.reshape(moved_shape) 

        dLdZ = np.moveaxis(dLdZ, -1, self.dim)
        return dLdZ

 

    