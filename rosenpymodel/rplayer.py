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
from rputils import actFunc, initFunc, decayFunc

class Layer():
    """
    Specification for a layer to be passed to the Neural Network during construction.  This
    includes a variety of parameters to configure each layer based on its activation type.
    """
    
        
    #  The attributes of the Layer class are initialized
    def __init__(self, ishape, neurons, oshape=0, weights_initializer=initFunc.random_normal, 
                 bias_initializer=initFunc.random_normal, gamma_initializer=initFunc.rbf_default, 
                 sigma_initializer=initFunc.ones, activation=actFunc.tanh, reg_strength=0.0, 
                 lambda_init=0.1, weights_rate=0.001, biases_rate=0.001, gamma_rate=0.0, sigma_rate=0.0, cvnn=1, 
                 lr_decay_method=decayFunc.none_decay,  lr_decay_rate=0.0, lr_decay_steps=1, kernel_initializer=initFunc.opt_ptrbf_weights,
                 kernel_size=3, module=None, category=1,layer_type="Fully"):
        
        """ 
        The __init__ method is the constructor of the Layer class. 
        
        Parameters
        ----------
            ishape: int
                The number of neurons in the first layer (the number of input features).  
            neurons: int
                The number of neurons in the hidden layer. 
                
            oshape: int
                The oshape is a specific argument for the RBF networks; in shallow CVNNs, 
                as there is only one layer, the input and output dimensions and the number 
                of hidden neurons must be specified when adding the layer.
                
            weights_initializer: str
                It defines the way to set the initial random weights, as a string. 
                
            bias_initializer: str 
                It defines the way to set the initial random biases, as string.
                
            gamma_initializer: str, optional
                It defines the way to set the initial random gamma, as string.
                
            sigma_initializer: str, optional
                It defines the way to set the initial sigma biases, as string. Initialization
                methods were defined in the file rp_utils.initFunc.
                
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
                
            activation: str
                Select which activation function this layer should use, as a string.
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
                
            reg_strength: float, optional
                It sets the regularization strength. The default value is 0.0, which means
                that regularization is turned off.
                
            lambda_init: float, optional
                It is the initial regularization factor strength.
                
            gamma_rate: float, optional
                The learning rate of matrix of the center vectors (RBF networks).
                
            sigma_rate: float, optional
                The learning rate of the vector of variance (RBF networks).
            
            cvnn: int
                It Defines which complex neural network the layer belongs to.
                
                * 1: CVFFNN or SCFFNN
                * 2: CVRBFNN
                * 3: FCRBFNN
                * 4: PTRBFNN
            module: str
                CuPy/Numpy module. This parameter is set at the time of 
                initialization of the NeuralNetwork class.
            kernel_initializer: str 
                It defines the way to set the initial, as string. Initialization
                methods were defined in the file rp_utils.initFunc.
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
        self.input = None
     
        self.reg_strength = reg_strength
        self.lambda_init = lambda_init
     
        self._activ_in, self._activ_out = None, None
        
        self.lr_decay_method=lr_decay_method  
        self.lr_decay_rate=lr_decay_rate
        self.lr_decay_steps=lr_decay_steps
        
        self.neurons = neurons
        self.oshape = oshape
        self.seuc = None
        self.phi = None
        self.kern = None
        
        self.layer_type = layer_type
        
        
        ## It initializes parameters for feedforward (FF) networks (CVFFNN and SCFFNN). 
        ## This includes initializing weights, biases, activation
        if cvnn==1:
            self.learning_rates = [weights_rate, biases_rate]
            self.weights = weights_initializer(module, ishape, neurons)
            self.biases = bias_initializer(module, 1, neurons)
            self.activation = activation
            
            self.ut = self.mt = self.vt = [initFunc.zeros(module, ishape, neurons), initFunc.zeros(module, 1, neurons)]
        ## It initializes parameters for CVRBFNN. 
        ## This includes initializing weights, biases, gamma and sigma 
        elif cvnn==2:
            self.learning_rates = [weights_rate, biases_rate, gamma_rate, sigma_rate]
            self.weights = weights_initializer(module, neurons, oshape, i=ishape)
            self.biases = bias_initializer(module, 1, oshape)
            
            self.gamma = gamma_initializer(module, neurons, ishape) 
            self.sigma = sigma_initializer(module, 1, neurons)
            
            self.ut = self.mt = self.vt = [initFunc.zeros(module, neurons, oshape), initFunc.zeros(module, 1, oshape), initFunc.zeros(module, 1, neurons), initFunc.zeros(module, neurons, ishape)]    
        ## It initializes parameters for FCRBFNN. 
        ## This includes initializing weights, biases, gamma and sigma 
        elif cvnn==3:
            self.learning_rates = [weights_rate, biases_rate, gamma_rate, sigma_rate]
            self.weights = weights_initializer(module, neurons, oshape)
            self.biases = bias_initializer(module, 1, oshape)
        
            self.gamma = gamma_initializer(module, neurons, ishape) #gpu.get_module().random.randint(2, size=[neurons,ishape])*0.7 + 1j*(gpu.get_module().random.randint(2, size=[neurons,ishape])*2-1)*0.7
            self.sigma = sigma_initializer(module, neurons, ishape) #gpu.get_module().random.randint(2, size=[neurons,ishape])*0.7 + 1j*(gpu.get_module().random.randint(2, size=[neurons,ishape])*2-1)*0.7
           
            self.ut = self.mt = self.vt = [initFunc.zeros(module, neurons, oshape), initFunc.zeros(module, 1, oshape), initFunc.zeros(module, neurons, ishape), initFunc.zeros(module, neurons, ishape)]
            
        ## It initializes parameters for PTRBFNN. 
        ## This includes initializing weights, biases, gamma and sigma     
        elif cvnn==4 and self.layer_type=="Fully":
            self.learning_rates = [weights_rate, biases_rate, gamma_rate, sigma_rate]
            self.weights = weights_initializer(module, neurons, oshape, i=ishape)
            self.biases = bias_initializer(module, 1, oshape)
            self.gamma =  gamma_initializer(module, neurons, ishape) #gpu.get_module().random.randint(2, size=[neurons,ishape])*0.7 + 1j*(gpu.get_module().random.randint(2, size=[neurons,ishape])*2-1)*0.7
            self.sigma = sigma_initializer(module, 1, neurons)
            
            self.ut = self.mt = self.vt = [initFunc.zeros(module, neurons, oshape), initFunc.zeros(module, 1, oshape), initFunc.zeros(module, 1, neurons), initFunc.zeros(module, neurons, ishape)]
            
            
        ## It initializes parameters for PTRBFNN. 
        ## This includes initializing weights, biases, gamma and sigma     
        elif cvnn==4 and self.layer_type=="Conv":
            self.category = category
            #self.oshape = (kernel_size + neurons - 1) if self.category == 1 else kernel_size
            
            self.oshape = kernel_size + neurons - 1 if self.category == 1 else kernel_size - neurons + 1 if kernel_size > neurons else kernel_size
            self.learning_rates = [weights_rate, biases_rate, gamma_rate, sigma_rate]
            self.weights = weights_initializer(module, 1, kernel_size, i=ishape)
            self.biases = bias_initializer(module, 1, kernel_size + neurons - 1 if self.category == 1 else kernel_size - neurons + 1 if kernel_size > neurons else kernel_size)
            self.gamma =  gamma_initializer(module, neurons, ishape) #gpu.get_module().random.randint(2, size=[neurons,ishape])*0.7 + 1j*(gpu.get_module().random.randint(2, size=[neurons,ishape])*2-1)*0.7
            self.sigma = sigma_initializer(module, 1, neurons)
            
            #self.kernel = kernel_initializer(module, 1, kernel_size)
            self.kernel_size = kernel_size 
            
            self.ut = self.mt = self.vt = [initFunc.zeros(module, 1, kernel_size), initFunc.zeros(module, 1, kernel_size + neurons - 1 if self.category == 1 else kernel_size - neurons + 1 if kernel_size > neurons else kernel_size), initFunc.zeros(module, 1, neurons), initFunc.zeros(module, neurons, ishape)]
            
            self.C = None
            
            
    
       
       