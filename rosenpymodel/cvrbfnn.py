
# -*- coding: utf-8 -*-
"""**RosenPy: An Open Source Python Framework for Complex-Valued Neural Networks**.
*Copyright © A. A. Cruz, K. S. Mayer, D. S. Arantes*.

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
from rputils import  regFunc, initFunc, decayFunc
from rosenpymodel import rplayer, rpnn

class CVRBFNN(rpnn.NeuralNetwork):
    """
    Specification for the Complex Valued Radial Basis Function Neural Network to be passed 
    to the model in construction.
    This includes the feedforward, backpropagation and adding layer methods specifics.
    
    This class derives from NeuralNetwork class.    
    """
    def feedforward(self, input_data):
        """
        Performs the feedforward operation on the neural network.

        Parameters:
        -----------
        input_data : array-like
            The input data to be fed into the neural network.

        Returns:
        --------
        array-like
            The output of the neural network after the feedforward operation.
        """
        
        # Check if GPU acceleration is enabled
        if self.gpu_enable:
            return self._feedforward_gpu(input_data)
        else:
            return self._feedforward_cpu(input_data)

    def _feedforward_gpu(self, x):
        """
        This method returns the output of the network if ``x`` is input.
        
        Parameters
        ----------
            x: array-like, shape (n_batch, n_inputs)
            
            Training vectors as real numbers, where n_batch is the
            batch and n_inputs is the number of input features.
        
        Returns
        -------
              y_pred: array-like, shape (n_batch, n_outputs) 
              
              The output of the last layer.
        
        """
        # Transfer data to GPU
        x_gpu = self.xp.array(x)
        
        # Set layer input for the first layer
        self.layers[0].input = self.xp.transpose(self.xp.tile(x, (self.layers[0].neurons, 1, 1)), axes=[1, 0, 2])
        
        # Calculate the distance between the input point and each center of the radial basis function
        self.layers[0].kern = input_layer - self.xp.tile(self.layers[0].gamma, (self.layers[0].input.shape[0], 1, 1))
        
        # Calculate the squared Euclidean distance separately for the real and imaginary components
        self.layers[0].seuc = self.xp.sum(self.xp.abs(self.layers[0].kern**2), axis=2) / self.layers[0].sigma
        
        # Activation measure for the neurons in the RBF layer
        self.layers[0].phi = self.xp.exp(-self.layers[0].seuc)
        
        # Calculate the output of the first layer
        self.layers[0].activ_out = self.xp.dot(phi, self.layers[0].weights) + self.layers[0].biases
        
        # Feedforward pass through the remaining layers
        for i in range(1, len(self.layers)):
            # Set layer input
            self.layers[i].input = self.xp.transpose(self.xp.tile(self.layers[0].activ_out, (self.layers[i].neurons, 1, 1)), axes=[1, 0, 2])
            
            # Calculate the distance between the input point and each center of the radial basis function
            self.layers[i].kern = self.layers[i].input - self.xp.tile(self.layers[i].gamma, (self.layers[i].input.shape[0], 1, 1))
            
            # Calculate the squared Euclidean distance separately for the real and imaginary components
            self.layers[i].seuc = self.xp.sum(self.xp.abs(self.layers[i].kern**2), axis=2) / self.layers[i].sigma
            
            # Activation measure for the neurons in the RBF layer
            self.layers[i].phi= self.xp.exp(-self.layers[i].seuc)
            
            # Calculate the output of the layer
            activ_out = self.xp.dot(self.layers[i].phi, self.layers[i].weights) + self.layers[i].biases
        
        # Transfer output back to CPU
        return self.layers[-1].activ_out.get()

    def _feedforward_cpu(self, x):
        """
        This method returns the output of the network if ``x`` is input.
        
        Parameters
        ----------
            x: array-like, shape (n_batch, n_inputs)
            
            Training vectors as real numbers, where n_batch is the
            batch and n_inputs is the number of input features.
        
        Returns
        -------
              y_pred: array-like, shape (n_batch, n_outputs) 
              
              The output of the last layer.
        
        """

        # Set layer input
        self.layers[0].input = self.xp.transpose(self.xp.tile(x, (self.layers[0].neurons,1,1)), axes=[1, 0, 2])
        
        # Calculate the the distance between the input point and each center of the radial basis function
        self.layers[0].kern = self.layers[0].input - self.xp.tile(self.layers[0].gamma, (self.layers[0].input.shape[0], 1,1))
        
        # Calculate the squared Euclidean distance separately for the real and imaginary components
        self.layers[0].seuc = (self.xp.sum(self.xp.abs(self.layers[0].kern**2), axis=2))/self.layers[0].sigma
      
        # Activation measure for the neurons in the RBF layer, based on the proximity of the input point to the centers of the radial basis functions
        self.layers[0].phi = self.xp.exp(-self.layers[0].seuc ) 
        
        # Calculate the output of the layer
        self.layers[0]._activ_out = (self.xp.dot(self.layers[0].phi, self.layers[0].weights) + self.layers[0].biases)
        
        for i in range(1, len(self.layers)):
            # Set layer input
            self.layers[i].input = self.xp.transpose(self.xp.tile(self.layers[i - 1]._activ_out, (self.layers[i].neurons,1,1)), axes=[1, 0, 2])
            
            # Calculate the the distance between the input point and each center of the radial basis function
            self.layers[i].kern = self.layers[i].input - self.xp.tile(self.layers[i].gamma, (self.layers[i].input.shape[0], 1,1))
            
            # Calculate the squared Euclidean distance separately for the real and imaginary components
            self.layers[i].seuc = (self.xp.sum(self.xp.abs(self.layers[i].kern**2), axis=2))/self.layers[i].sigma
          
            # Activation measure for the neurons in the RBF layer, based on the proximity of the input point to the centers of the radial basis functions
            self.layers[i].phi = self.xp.exp(-self.layers[i].seuc ) 
            
            # Calculate the output of the layer
            self.layers[i]._activ_out = (self.xp.dot(self.layers[i].phi, self.layers[i].weights) + self.layers[i].biases)
            
        # Return the output of the last layer
        return self.layers[-1]._activ_out
    
    
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
        if self.gpu_enable:
            return self._backprop_gpu(y, y_pred, epoch)
        else:
            return self._backprop_cpu(y, y_pred, epoch) 
        
    def _backprop_cpu(self, y, y_pred, epoch):
        """
        This class provids a way to calculate the gradients of a target class output.

        Parameters
        ----------
        y : array-like, shape (n_samples, n_outputs)
            Target values are real numbers representing the desired outputs.
        y_pred : array-like, shape (n_samples, n_outputs)
            Target values are real numbers representing the predicted outputs.
        epoch : int
            Current number of the training epoch for updating the smoothing factor. 

        Returns
        -------
        None.

        """
        error = y - y_pred
        last = True
        auxK = aux = 0
        
        for layer in reversed(self.layers):
            psi = error if last else -self.xp.sum(self.xp.matmul(self.xp.transpose(auxK, (0, 2, 1)), aux[:, :, self.xp.newaxis]), axis=2)
            
            last = False
            auxK = layer.kern        
           
            epsilon = self.xp.dot(psi.real, (layer.weights.real.T)) + self.xp.dot(psi.imag, (layer.weights.imag.T))
          
            beta = layer.phi/layer.sigma

            aux  = self.xp.multiply(epsilon, beta)  
            
            # Compute the regularization l2
            regl2 = (regFunc.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch))

            grad_w = self.xp.dot(layer.phi.T, psi) - (regl2 if layer.reg_strength else 0)*layer.weights
            grad_b = self.xp.divide(sum(psi), psi.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.biases
            
            s_a = self.xp.multiply(aux, layer.seuc)
            grad_s = self.xp.divide(sum(s_a), s_a.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.sigma

            g_a = self.xp.multiply(aux[:, :, self.xp.newaxis], layer.kern)
            grad_g = self.xp.divide(sum(g_a), g_a.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.gamma
            
            
            #self.learning_rate = self.lr_decay_method(self.lr_initial, epochs, self.lr_decay_rate, self.lr_decay_steps)
            
            layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut  = self.optimizer.update_parameters([layer.weights, layer.biases, layer.sigma, layer.gamma], 
                                                         [grad_w, grad_b, grad_s, grad_g], 
                                                         layer.learning_rates, 
                                                         epoch, layer.mt, layer.vt, layer.ut)
            
            layer.sigma = self.xp.where(layer.sigma.real>0.0001, layer.sigma.real, 0.0001) 
    
    def _backprop_gpu(self, y, y_pred, epoch):
            """
            This class provides a way to calculate the gradients of a target class output.
        
            Parameters
            ----------
            y : array-like, shape (n_samples, n_outputs)
                Target values are real numbers representing the desired outputs.
            y_pred : array-like, shape (n_samples, n_outputs)
                Target values are real numbers representing the predicted outputs.
            epoch : int
                Current number of the training epoch for updating the smoothing factor. 
        
            Returns
            -------
            None.
        
            """
            error = y - y_pred
            last = True
            auxK = aux = 0
            
            
            
            # Loop through the layers in reverse order
            for layer in reversed(self.layers):
                psi = error if last else -self.xp.sum(self.xp.matmul(self.xp.transpose(auxK, (0, 2, 1)), aux[:, :, self.xp.newaxis]), axis=2)
                
                last = False
                auxK = layer.kern        
                
                # Create a CUDA stream
                stream = self.xp.cuda.Stream()
                
                epsilon_real = self.xp.dot(psi.real, layer.weights.real.T)
                epsilon_imag = self.xp.dot(psi.imag, layer.weights.imag.T)
                
                # Element-wise multiplication of epsilon and beta
                epsilon_real_beta = epsilon_real * layer.phi / layer.sigma
                epsilon_imag_beta = epsilon_imag * layer.phi / layer.sigma
                
                # Compute the regularization l2
                regl2 = (regFunc.l2_regularization(cp, layer.lambda_init, layer.reg_strength, epoch))
                
               
                # Update weights, biases, sigma and gamma asynchronously using streams
                with stream:
                
                    # Compute gradients
                    grad_w = self.xp.dot(layer.phi.T, psi) - (regl2 if layer.reg_strength else 0)*layer.weights
                    grad_b = self.xp.mean(psi) - (regl2 if layer.reg_strength else 0)*layer.biases
                
                    s_a = self.xp.multiply(aux, layer.seuc)
                    grad_s = self.xp.mean(s_a) - (regl2 if layer.reg_strength else 0)*layer.sigma
    
                    g_a = self.xp.multiply(aux[:, :, self.xp.newaxis], layer.kern)
                    grad_g = self.xp.mean(g_a) - (regl2 if layer.reg_strength else 0)*layer.gamma
                
            
        
                # Update parameters using the optimizer
                layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut  = self.optimizer.update_parameters([layer.weights, layer.biases, layer.sigma, layer.gamma], 
                                                         [grad_w, grad_b, grad_s, grad_g], 
                                                         layer.learning_rates, 
                                                         epoch, layer.mt, layer.vt, layer.ut)
                
                # Ensure sigma doesn't become too small
                layer.sigma = self.xp.where(layer.sigma.real > 0.0001, layer.sigma.real, 0.0001)
    
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
        return ((input_data - mean) / std_dev) * (1/self.xp.sqrt(2*input_data.shape[1]))
    

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
        return (normalized_output_data / (1/self.xp.sqrt(2*normalized_output_data.shape[1]))) * std_dev + mean


    
    def addLayer(self, neurons, ishape=0, oshape=0, 
                     weights_initializer=initFunc.opt_crbf_weights, bias_initializer=initFunc.zeros, sigma_initializer=initFunc.ones_real, gamma_initializer=initFunc.opt_crbf_gamma,
                     weights_rate=0.001, biases_rate=0.001, gamma_rate=0.01, sigma_rate=0.01, 
                     reg_strength=0.0, lambda_init=0.1,
                     lr_decay_method=decayFunc.none_decay,  lr_decay_rate=0.0, lr_decay_steps=1,
                     module=None):
            
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
            module :str        
                CuPy/Numpy module. This parameter is set at the time of 
                initialization of the NeuralNetwork class.
                
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
                                          cvnn=2, 
                                          lr_decay_method=lr_decay_method,  
                                          lr_decay_rate=lr_decay_rate, 
                                          lr_decay_steps=lr_decay_steps,
                                          module=self.xp))