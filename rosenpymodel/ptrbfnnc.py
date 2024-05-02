# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:21:31 2024

@author: User
"""

"""**RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks**.
*Copyright Â© A. A. Cruz, K. S. Mayer, D. S. Arantes*.

*License*

This file is part of RosenPy.
RosenPy is an open source framework distributed under the terms of the GNU General 
Public License, as published by the Free Software Foundation, either version 3 of 
the License, or (at your option) any later version. For additional information on 
license terms, please open the Readme.md file.

RosenPy is distributed in the hope that it will be useful to every user, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License
along with RosenPy.  If not, see <http://www.gnu.org/licenses/>.
"""
from rputils import regFunc, initFunc, decayFunc
from rosenpymodel import rpoptimizer as opt
from rosenpymodel import rplayer, rpnn

class PTRBFNNConv(rpnn.NeuralNetwork):   
    """
    Specification for the Deep Phase Transmittance Radial Basis Function Neural Network 
    to be passed to the model in construction.
    This includes the feedforward, backpropagation and adding layer methods specifics.
    
    This class derives from NeuralNetwork class.    
    """    
    
    def _matrixC(self, phi, w, tp):  
        """
        Generates the coupling matrix C responsible for converting the linear combination 
        from the fully connected operation into a convolutional operation.

        Parameters:
        -----------
        phi : array-like
            Array containing the values of the basis functions.
        w : array-like
            Array containing the values of the weights.
        tp : int
            Type of transformation (Transient and steady-state: 1; or steady-state: 0).

        Returns:
        --------
        array-like
            The coupling matrix C.
        """
        # List to store matrices C
        C_list = []
        
        if tp==1: # Transient and steady-state
            for phi_row in range(phi.shape[0]):  # Iterate over the rows of phi
                # # Create a matrix C for this row of phi
                m = phi.shape[1]  # Number of columns of phi
                n = w.shape[1]    # Number of rows of w
               
                C = self.xp.array([[phi[phi_row, row - column] if row - column >= 0 and row - column < m else 0 for column in range(n) ] for row in range(m + n - 1)])
                # Add the matrix C to the list
                C_list.append(C)
        else: # steady-state     
        # Stack the matrices C along the first dimension to form a tensor
            for phi_row in range(phi.shape[0]):  # Iterate over the rows of phi
                # # Create a matrix C for this row of phi
                m = phi.shape[1]  # Number of columns of phi
                n = w.shape[1]    # Number of rows of w
                if m > n:
                    C =  self.xp.array([[phi[phi_row, row+n-1-column] for column in range(n)] for row in range(m-n+1)])
                else:
                    C =  self.xp.array([[phi[phi_row, n-m-column+row-1] if n - m - column + row - 1 >= 0 and n - m - column + row - 1 < m else 0 for column in range(n)] for row in range(n-m+1)])
                C_list.append(C)
                
        C_tensor = self.xp.stack(C_list)
            
        return C_tensor
        
    def _matrizK(self, phi, w, tp):    
        """
        Generates the coupling matrix K responsible for transforming between transient and steady-state or steady-state operations.

        Parameters:
        -----------
        phi : array-like
            Array containing the values of the basis functions.
        w : array-like
            Array containing the values of the weights.
        tp : int
            Type of transformation (Transient and steady-state: 1; or steady-state: 0).

        Returns:
        --------
        array-like
            The coupling matrix K.
        """       
        # List to store matrices C
        K_list = []
        if tp==1: # Transient and steady-state
            for phi_row in range(phi.shape[0]):  # Iterate over the rows of phi
                m = len(phi[0])
                n = len(w[0])
                
                K = self.xp.zeros((m, m + n - 1), dtype=self.xp.complex128)
                
                for row in range(n):
                    for column in range(m + n - 1):
                        if column - row >= 0 and column - row < len(w[0]):
                            K[row, column] = w[0, column - row]
                
          
            
                # Add the matrix K to the list
                K_list.append(K)
        else: # steady-state 
            for phi_row in range(phi.shape[0]):  # Iterate over the rows of phi
                # # Create a matrix C for this row of phi
                m = phi.shape[1]  # Number of columns of phi
                n = w.shape[1]    # Number of rows of w
                if m > n:
                    K = self.xp.array([[w[0, column-row+n-1] if column - row + n - 1 < n and column - row + n - 1 >= 0 else 0 for column in range(m-n+1)] for row in range(m)])
                else:
                    K = self.xp.array([[w[0, m - row + column -1] if m - row + column - 1>= 0 and m - row + column -1 < n else 0 for column in range(n-m+1)] for row in range(m)])
                
                # Add the matrix K to the list
                K_list.append(K)
            
        # Stack the matrices C along the first dimension to form a tensor
        K_tensor = self.xp.stack(K_list)
            
        return K_tensor
    

    def _fully_feedforward(self, y_pred, layer):
        """
        Performs the feedforward operation specific to a fully connected layer.

        Parameters:
        -----------
        y_pred : array-like
            The input data to be fed into the fully connected layer.
        layer : FullyConnectedLayer
            The fully connected layer object.

        Returns:
        --------
        array-like
            The output of the fully connected layer after the feedforward operation.
        """
        
        layer.kern = y_pred[:, self.xp.newaxis, :].repeat(layer.neurons, axis=1) - layer.gamma
        
        layer.seuc = self.xp.sum(layer.kern.real ** 2, axis=2) / layer.sigma.real + 1j * (self.xp.sum(layer.kern.imag ** 2, axis=2) / layer.sigma.imag)
        
        layer.phi = self.xp.exp(-layer.seuc.real) + 1j * self.xp.exp(-layer.seuc.imag)
        
        layer._activ_out = self.xp.dot(layer.phi, layer.weights) + layer.biases
        
        return layer._activ_out 
        
    def _conv_feedforward_tp(self, x, layer):
        """
        Performs the feedforward operation specific to a convolutional layer.

        Parameters:
        -----------
        x : array-like
            The input data to be fed into the convolutional layer.
        layer : ConvLayer
            The convolutional layer object.

        Returns:
        --------
        array-like
            The output of the convolutional layer after the feedforward operation.
        """
        
        layer.input = self.xp.transpose(self.xp.tile(x, (layer.neurons,1,1)), axes=[1, 0, 2])
                
        # Calculate the the distance between the input point and each center of the radial basis function
        layer.kern = layer.input - self.xp.tile(layer.gamma, (layer.input.shape[0], 1,1))
        
        # Calculate the squared Euclidean distance separately for the real and imaginary components
        aux_r = self.xp.sum(layer.kern.real*layer.kern.real, axis=2)
        aux_i = self.xp.sum(layer.kern.imag*layer.kern.imag, axis=2)
        
        seuc_r = aux_r/layer.sigma.real
        seuc_i = aux_i/layer.sigma.imag
        
        layer.seuc = seuc_r + 1j*seuc_i
        
        # Activation measure for the neurons in the RBF layer, based on the proximity of the input point to the centers of the radial basis functions
        layer.phi = self.xp.exp(-seuc_r) + 1j*(self.xp.exp(-seuc_i))
        
        layer.C = self._matrixC(layer.phi, layer.weights, layer.category)
        
        # Calculate the output of the layer      
        aux = self.xp.dot(layer.weights, self.xp.transpose(layer.C, (0, 2, 1)))
        
        layer._activ_out = self.xp.squeeze(aux) + layer.biases

        return layer._activ_out
    
    
    def feedforward(self, x):
        """
        Performs the feedforward operation on the neural network.

        Parameters:
        -----------
        x : array-like
            The input data to be fed into the neural network.

        Returns:
        --------
        array-like
            The output of the neural network after the feedforward operation.
        """
        conv_layer_found = False
        fully_connected_found = False
    
        # Iterate through the layers of the neural network
        for i, layer in enumerate(self.layers):
            # Check if the current layer is a convolutional layer
            if layer.layer_type == "Conv":
                conv_layer_found = True
                # Perform convolutional feedforward operation
                x = self._conv_feedforward_tp(x, layer)
    
            # Check if the current layer is a fully connected layer
            elif layer.layer_type == "Fully":
                # If there are no convolutional layers before, perform fully connected feedforward operation
                if not conv_layer_found:
                    fully_connected_found = True
                else:
                    # If there are convolutional layers before, the current layer must be fully connected
                    if fully_connected_found:
                        raise ValueError("If there are convolutional layers, the last layer must be fully connected.")
                x = self._fully_feedforward(x, layer)
    
        return x


    def backprop(self, y, y_pred, epoch):
        """
        Performs the backpropagation operation on the neural network.

        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values from the neural network.
        epoch : int
            The current epoch number.

        Returns:
        --------
        array-like
            The gradients of the loss function with respect to the network parameters.
        """

        # Check if GPU acceleration is enabled
        last = True
        auxK = aux_r = aux_i = 0
        
        for layer in reversed(self.layers):
            if layer.layer_type == "Conv":
                auxK, last, aux_r, aux_i = self._conv_backprop_tp(y, y_pred, epoch, layer, auxK, last, aux_r, aux_i)
    
            elif layer.layer_type == "Fully":
                auxK, last, aux_r, aux_i = self._fully_backprop(y, y_pred, epoch, layer, auxK, last, aux_r, aux_i)
    
    def _conv_backprop_tp(self, y, y_pred, epoch, layer, auxK, last, aux_r, aux_i):
        """
        Performs the backpropagation operation specific to a convolutional layer.

        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values from the convolutional layer.
        epoch : int
            The current epoch number.
        layer : ConvLayer
            The convolutional layer object.
        auxK : array-like
            A kernel from the previous layer, which is obtained by subtracting the input by the gamma.
        last : bool
            Flag indicating if the current layer is the last layer in the network.
        aux_r : array-like
            Array containing the real part resulting from the multiplication of epsilon by phi under sigma.
        aux_i : array-like
            Array containing the imaginary part resulting from the multiplication of epsilon by phi under sigma.
        Returns:
        --------
        tuple
            A tuple containing the values to be used in the calculations of the following layers.
        """
        
        psi = -self.xp.sum(self.xp.matmul(self.xp.transpose(auxK.real, (0, 2, 1)), aux_r[:, :, self.xp.newaxis]) + 1j * self.xp.matmul(self.xp.transpose(auxK.imag, (0, 2, 1)), aux_i[:, :, self.xp.newaxis]), axis=2)
        
        auxK = layer.kern
    
        K = self._matrizK(layer.phi, layer.weights, layer.category)
        epsilon =  self.xp.einsum('ij,ikj->ik', psi, self.xp.conj(K)) ##resultado é correto?
        
        _psi = self.xp.transpose(self.xp.expand_dims(psi, axis=-1), axes=[1,0,2])
        
        beta_r = layer.phi.real / layer.sigma.real
        beta_i = layer.phi.imag / layer.sigma.imag

        aux_r = epsilon.real * beta_r
        aux_i = epsilon.imag * beta_i

        # Precompute regularization term
        regl2 = (regFunc.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch))

        
        grad_w = self.xp.tensordot(_psi, self.xp.conj(layer.C), axes=([0, 1],[1, 0]))/layer.C.shape[0] - (regl2 if layer.reg_strength else 0) * layer.weights
        grad_b = self.xp.mean(psi, axis=0) - (regl2 if layer.reg_strength else 0) * layer.biases
            
        s_a = self.xp.multiply(aux_r, layer.seuc.real) + 1j * self.xp.multiply(aux_i, layer.seuc.imag)
        grad_s =  self.xp.mean(s_a, axis=0) - (regl2 if layer.reg_strength else 0) * layer.sigma


        g_a = self.xp.multiply(aux_r[:, :, self.xp.newaxis],layer.kern.real) + 1j*self.xp.multiply(aux_i[:, :, self.xp.newaxis],layer.kern.imag)
        grad_g = self.xp.mean(g_a, axis=0) - (regl2 if layer.reg_strength else 0) * layer.gamma

                 
        layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut  = self.optimizer.update_parameters([layer.weights, layer.biases, layer.sigma, layer.gamma], 
                                                     [grad_w, grad_b, grad_s, grad_g], 
                                                     layer.learning_rates, 
                                                     epoch, layer.mt, layer.vt, layer.ut)
        


        # Clip sigma to avoid small values
        layer.sigma = self.xp.maximum(layer.sigma, 0.0001)        
        
        return auxK, last, aux_r, aux_i
    

    def _fully_backprop(self, y, y_pred, epoch, layer, auxK, last, aux_r, aux_i):
        """
        Performs the backpropagation operation specific to a fully layer.

        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values from the fully layer.
        epoch : int
            The current epoch number.
        layer : ConvLayer
            The convolutional layer object.
        auxK : array-like
            A kernel from the previous layer, which is obtained by subtracting the input by the gamma.
        last : bool
            Flag indicating if the current layer is the last layer in the network.
        aux_r : array-like
            Array containing the real part resulting from the multiplication of epsilon by phi under sigma.
        aux_i : array-like
            Array containing the imaginary part resulting from the multiplication of epsilon by phi under sigma.
        Returns:
        --------
        tuple
            A tuple containing the values to be used in the calculations of the following layers.
        """
        
        error = (y - y_pred)
        psi = error if last else -self.xp.sum(self.xp.matmul(self.xp.transpose(auxK.real, (0, 2, 1)), aux_r[:, :, self.xp.newaxis]) + 1j * self.xp.matmul(self.xp.transpose(auxK.imag, (0, 2, 1)), aux_i[:, :, self.xp.newaxis]), axis=2)
        last = False
        auxK = layer.kern
    
        epsilon = self.xp.dot(psi, self.xp.conj(layer.weights.T))
        beta_r = layer.phi.real / layer.sigma.real
        beta_i = layer.phi.imag / layer.sigma.imag

        aux_r = epsilon.real * beta_r
        aux_i = epsilon.imag * beta_i

        # Precompute regularization term
        regl2 = (regFunc.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch))

        grad_w = self.xp.dot(self.xp.conj(layer.phi.T), psi) - (regl2 if layer.reg_strength else 0) * layer.weights
        grad_b = self.xp.mean(psi, axis=0) - (regl2 if layer.reg_strength else 0) * layer.biases
            
        s_a = self.xp.multiply(aux_r, layer.seuc.real) + 1j * self.xp.multiply(aux_i, layer.seuc.imag)
        grad_s =  self.xp.mean(s_a, axis=0) - (regl2 if layer.reg_strength else 0) * layer.sigma

        g_a = self.xp.multiply(aux_r[:, :, self.xp.newaxis], layer.kern.real) + 1j * self.xp.multiply(aux_i[:, :, self.xp.newaxis], layer.kern.imag)
        grad_g = self.xp.mean(g_a, axis=0) - (regl2 if layer.reg_strength else 0) * layer.gamma

                 
        layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut  = self.optimizer.update_parameters([layer.weights, layer.biases, layer.sigma, layer.gamma], 
                                                     [grad_w, grad_b, grad_s, grad_g], 
                                                     layer.learning_rates, 
                                                     epoch, layer.mt, layer.vt, layer.ut)
        


        # Clip sigma to avoid small values
        layer.sigma = self.xp.maximum(layer.sigma, 0.0001)        
        
        return auxK, last, aux_r, aux_i
    
    # Method to normalize the data
    def normalize_data(self, input_data, mean, std_dev):
        """
        Normalize the input data.

        Args:
            input_data (cupy/numpy.ndarray): Input data to be normalized.

        Returns:
            cupy/numpy.ndarray: Normalized input data.
        """
        # Normalize the  data using mean and standard deviation and scale by the square root of the input data shape
        return ((input_data - mean) / std_dev) * (1 / self.xp.sqrt(input_data.shape[1]))


    # Method to denormalize the output data
    def denormalize_outputs(self, normalized_output_data, mean, std_dev):
        """
        Denormalize the output data.

        Args:
            normalized_output_data (cupy/numpy.ndarray): Normalized output data to be denormalized.
            
        Returns:
            cupy/numpy.ndarray: Denormalized output data.
        """
        # Denormalize the output data using mean and standard deviation and scale by the inverse of the square root of the original output data shape
        return (normalized_output_data * std_dev) / (1 / self.xp.sqrt(normalized_output_data.shape[1])) + mean

    
    def addLayer(self, neurons, ishape=0, oshape=0, weights_initializer=initFunc.opt_ptrbf_weights, bias_initializer=initFunc.zeros, 
                 sigma_initializer=initFunc.ones, gamma_initializer=initFunc.opt_ptrbf_gamma,
                 reg_strength=0.0, lambda_init=0.1, weights_rate=0.001, biases_rate=0.001, gamma_rate=0.01, sigma_rate=0.01, 
                 lr_decay_method=decayFunc.none_decay,  lr_decay_rate=0.0, lr_decay_steps=1,
                 kernel_initializer=initFunc.opt_ptrbf_gamma, kernel_size=3,
                 module=None, category=1,
                 layer_type="Fully"):
        """
        The method is responsible for adding the layers to the neural network.

        Parameters
        ----------
        neurons : int
            The number of neurons in the hidden layer. If the ishape is different zero and 
            it is the first layer of the model, neurons represents the number of neurons 
            in the first layer (the number of input features).
        ishape : int, optional
            The number of neurons in the first layer (the number of input features). The default is 0.
        weights_initializer : str, optional
            It defines the way to set the initial random weights, as a string. The default is initFunc.random_normal.
        bias_initializer : str, optional
            It defines the way to set the initial random biases, as a string. The default is initFunc.random_normal.
            Initialization methods were defined in the file rp_utils.initFunc.
            
            * rp_utils.initFunc.zeros
            * rp_utils.initFunc.ones
            * rp_utils.initFunc.ones_real
            * rp_utils.initFunc.random_normal
            * rp_utils.initFunc.random_uniform
            * rp_utils.initFunc.glorot_normal
            * rp_utils.initFunc.glorot_uniform
            * rp_utils.initFunc.rbf_default
            * rp_utils.initFunc.opt_crbf_weights
            * rp_utils.initFunc.opt_crbf_gamma
            * rp_utils.initFunc.opt_conv_ptrbf_weights
            * rp_utils.initFunc.opt_ptrbf_weights
            * rp_utils.initFunc.opt_ptrbf_gamma
            * rp_utils.initFunc.ru_weights_ptrbf
            * rp_utils.initFunc.ru_gamma_ptrbf
            
        activation : str, optional
            Select which activation function this layer should use, as a string. The default is actFunc.tanh.
            Activation methods were defined in the file rp_utils.actFunc.
            
            * rp_utils.actFunc.sinh
            * rp_utils.actFunc.atanh
            * rp_utils.actFunc.asinh
            * rp_utils.actFunc.tan
            * rp_utils.actFunc.sin
            * rp_utils.actFunc.atan
            * rp_utils.actFunc.asin
            * rp_utils.actFunc.acos
            * rp_utils.actFunc.sech
            
        reg_strength : float, optional
            It sets the regularization strength. The default is 0.0., which means that regularization is turned off
        lambda_init : float, optional
            It is the initial regularization factor strength. The default is 0.1.
        module : str        
            CuPy/Numpy module. This parameter is set at the time of 
            initialization of the NeuralNetwork class.
        kernel_size : int
            Size of the kernel of the convolutional layer
        category : int
            Type of convolution: transient and steady-state (1) or steady-state (0)
        layer_type : str
            Layer type: fully connected or convolutional - conv.
        Returns
        -------
        None.

        """
        self.layers.append(rplayer.Layer(ishape if not len(self.layers) else self.layers[-1].oshape, neurons, neurons if not oshape else oshape, 
                                          weights_initializer=weights_initializer, 
                                          bias_initializer=bias_initializer, 
                                          sigma_initializer=sigma_initializer, 
                                          gamma_initializer=gamma_initializer,
                                          reg_strength=reg_strength, 
                                          lambda_init=lambda_init,
                                          weights_rate=weights_rate,
                                          biases_rate=biases_rate,
                                          sigma_rate=sigma_rate,
                                          gamma_rate=gamma_rate,
                                          cvnn=4,             
                                          lr_decay_method=lr_decay_method,  
                                          lr_decay_rate=lr_decay_rate, 
                                          lr_decay_steps=lr_decay_steps,
                                          kernel_initializer=kernel_initializer,
                                          kernel_size=kernel_size,
                                          module=self.xp,
                                          category=category,
                                          layer_type=layer_type))
    
    