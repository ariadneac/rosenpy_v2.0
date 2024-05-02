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
from rputils import regFunc, initFunc, actFunc, decayFunc
from rosenpymodel import rplayer, rpnn
import numpy as np

class FCRBFNN(rpnn.NeuralNetwork):
    """
    Specification for the Fully Complex Transmittance Radial Basis Function Neural Network 
    to be passed to the model in construction.
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
      
        # Set layer input
        self.layers[0].input = self.xp.transpose(self.xp.tile(x, (self.layers[0].neurons, 1, 1)), axes=[1, 0, 2])

        # Calculate the distance between the input point and each center of the radial basis function
        tiled_gamma = self.xp.tile(self.layers[0].gamma, (self.layers[0].input.shape[0], 1, 1))
        self.layers[0].seuc = self.layers[0].input - tiled_gamma

        self.layers[0].kern = self.xp.sum(self.xp.multiply(self.layers[0].sigma, self.layers[0].seuc), axis=2)

        # Apply activation function
        self.layers[0].phi = actFunc.sech(self.xp, self.layers[0].kern)

        # Calculate the output of the layer
        weights_dot_phi = self.xp.tensordot(self.layers[0].phi, self.layers[0].weights, axes=([1], [0]))
        self.layers[0]._activ_out = weights_dot_phi + self.layers[0].biases

        # Return the output of layer
        return self.layers[-1]._activ_out

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
        self.layers[0].seuc = self.layers[0].input - self.xp.tile(self.layers[0].gamma, (self.layers[0].input.shape[0], 1,1))
        
        self.layers[0].kern = self.xp.sum(self.xp.multiply(self.layers[0].sigma, self.layers[0].seuc), axis=2)
        
        # Apply activation function
        self.layers[0].phi = actFunc.sech (self.xp, self.layers[0].kern)
        
        # Calculate the output of the layer
        self.layers[0]._activ_out = self.xp.dot(self.layers[0].phi, self.layers[0].weights) + self.layers[0].biases
        
        # Return the output of layer
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
        
    def _backprop_gpu(self, y, y_pred, epoch):
        """
        Performs the backpropagation operation on the neural network using GPU acceleration.

        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values from the neural network.
        epoch : int
            The current epoch number.
        """

        # Compute the error
        error = y - y_pred
    
        layer = self.layers[0]  
        
        phi_l = actFunc.sech(self.xp, layer.kern, derivative=True)
        
        a = self.xp.multiply(self.xp.dot(error, self.xp.conj(layer.weights).T), self.xp.conj(phi_l))
        
        regl2 = (regFunc.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch))
        
        with self.xp.cuda.Stream():  # Criando um novo stream
            grad_w = self.xp.dot(layer.phi.T, error) - (regl2 if layer.reg_strength else 0) * layer.weights
            grad_b = self.xp.mean(error, axis=0) - (regl2 if layer.reg_strength else 0) * layer.biases

            s_a = a[:, :, self.xp.newaxis] * self.xp.conj(layer.seuc)
            grad_s = self.xp.mean(s_a, axis=0) - (regl2 if layer.reg_strength else 0) * layer.sigma
            
            g_a = self.xp.multiply(a[:, :, self.xp.newaxis], self.xp.tile(self.xp.conj(layer.sigma), (a.shape[0], 1, 1)))
            grad_g = -self.xp.mean(g_a, axis=0) - (regl2 if layer.reg_strength else 0) * layer.gamma
        
            layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut  = self.optimizer.update_parameters([layer.weights, layer.biases, layer.sigma, layer.gamma], 
                                                         [grad_w, grad_b, grad_s, grad_g], 
                                                         layer.learning_rates, 
                                                         epoch, layer.mt, layer.vt, layer.ut)
        
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
        
        for layer in reversed(self.layers):
            phi_l = actFunc.sech(self.xp, layer.kern, derivative=True)
      
            a = self.xp.multiply(self.xp.dot(error, self.xp.conj(layer.weights).T), self.xp.conj(phi_l))
            
            # Compute the regularization l2
            regl2 = (regFunc.l2_regularization(self.xp, layer.lambda_init, layer.reg_strength, epoch))

            grad_w = self.xp.dot(layer.phi.T, error) - (regl2 if layer.reg_strength else 0)*layer.weights
                     
            grad_b = self.xp.divide(sum(error), error.shape[0])  - (regl2 if layer.reg_strength else 0)*layer.biases
                     
            s_a = a[:, :, self.xp.newaxis] * self.xp.conj(layer.seuc)
            grad_s = self.xp.divide(sum(s_a), s_a.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.sigma
            
            g_a = self.xp.multiply(a[:, :, self.xp.newaxis],self.xp.tile(self.xp.conj(layer.sigma), (a.shape[0],1,1)))
            grad_g = -self.xp.divide(sum(g_a), g_a.shape[0]) - (regl2 if layer.reg_strength else 0)*layer.gamma
            
            layer.weights, layer.biases, layer.sigma, layer.gamma, layer.mt, layer.vt, layer.ut  = self.optimizer.update_parameters([layer.weights, layer.biases, layer.sigma, layer.gamma], 
                                                         [grad_w, grad_b, grad_s, grad_g], 
                                                         layer.learning_rates, 
                                                         epoch, layer.mt, layer.vt, layer.ut)
    
    
    def addLayer(self, ishape, neurons, oshape, weights_initializer=initFunc.random_normal, bias_initializer=initFunc.ones, 
                 gamma_initializer=initFunc.rbf_default, sigma_initializer=initFunc.rbf_default, 
                 reg_strength=0.0, lambda_init=0.1, gamma_rate=0.01, sigma_rate=0.01,
                 lr_decay_method=decayFunc.none_decay,  lr_decay_rate=0.0, lr_decay_steps=1, module=None):
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
        self.layers.append(rplayer.Layer(ishape, neurons, oshape, 
                                          weights_initializer=weights_initializer, 
                                          bias_initializer=bias_initializer, 
                                          gamma_initializer=gamma_initializer, 
                                          sigma_initializer=gamma_initializer,
                                          reg_strength=reg_strength, 
                                          lambda_init=lambda_init, 
                                          sigma_rate=sigma_rate,
                                          gamma_rate=gamma_rate,
                                          cvnn=3,
                                          lr_decay_method=lr_decay_method,  
                                          lr_decay_rate=lr_decay_rate, 
                                          lr_decay_steps=lr_decay_steps,
                                          module=self.xp))
               