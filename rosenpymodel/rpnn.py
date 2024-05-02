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
from rputils import costFunc, decayFunc, batchGenFunc, gpu
from rosenpymodel import rpoptimizer as opt
import numpy as np

        
class NeuralNetwork():
    """
    Abstract base class for wrapping all neural network functionality from RosenPy. 
    This is a superclass. 
    """
   
    def __init__(self, cost_func=costFunc.mse, patience=10000000000000, gpu_enable=False):     
        """
        Initializes the neural network with default parameters.

        Parameters:
        -----------
        cost_func : function, optional
            The cost function to be used for training the neural network. Default is mean squared error (MSE).
        patience : int, optional
            The patience parameter for early stopping during training. Default is a large value to avoid early stopping.
        gpu_enable : bool, optional
            Flag indicating whether GPU acceleration is enabled. Default is False.
        """
        
        self.xp = gpu.module(gpu_enable) 
        self.gpu_enable = gpu_enable
        self.layers = []
        self.cost_func = cost_func
    
        self.optimizer = None
        self.patience, self.waiting = patience, 0
        
        self._best_model, self._best_loss = self.layers, self.xp.inf
        self._history = {'epochs': [], 'loss': [], 'loss_val': []}

        

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=100, verbose=10, batch_gen=batchGenFunc.batch_sequential, batch_size=1, optimizer=opt.GradientDescent(beta=100, beta1=0.9, beta2=0.999)):
        """
        Trains the neural network on the provided training data.

        Parameters:
        -----------
        x_train : array-like
            The input training data.
        y_train : array-like
            The target training data.
        x_val : array-like, optional
            The input validation data. Default is None.
        y_val : array-like, optional
            The target validation data. Default is None.
        epochs : int, optional
            The number of training epochs. Default is 100.
        verbose : int, optional
            Controls the verbosity of the training process. Default is 10.
        batch_gen : function, optional
            The batch generation function to use during training. Default is batchGenFunc.batch_sequential.
        batch_size : int, optional
            The batch size to use during training. Default is 1.
        optimizer : Optimizer, optional
            The optimizer to use during training. Default is GradientDescent with specified parameters.
        """
        self.verification(x_train)
        self.optimizer = optimizer
        self.optimizer.module(self.xp)
        
        x_train, y_train = self.convert_data(x_train), self.convert_data(y_train)

        self.mean_in = self.xp.mean(x_train)
        self.mean_out = self.xp.mean(y_train)

        self.std_in = self.xp.std(x_train)
        self.std_out = self.xp.std(y_train)
        
        # If validation data is not provided, the method uses the training data for validation
        x_val, y_val = (x_train, y_train) if (x_val is None or y_val is None) else (self.convert_data(x_val), self.convert_data(y_val)) 
        

        x_train, y_train = self.normalize_data(x_train, self.mean_in, self.std_in), self.normalize_data(y_train, self.mean_out, self.std_out) 
        x_val, y_val = self.normalize_data(x_val, self.mean_in, self.std_in), self.normalize_data(y_val, self.mean_out, self.std_out)
        
        for epoch in range(1,epochs+1):
            
            # It generates batches of training data using the specified batch generation function
            x_batch, y_batch = batch_gen(self.xp, x_train, y_train, batch_size)
            
            self.update_learning_rate(epoch)
            
            # For each batch, it performs feedforward and backpropagation to update the model's parameters
            for x_batch1, y_batch1 in zip(x_batch, y_batch):
                    y_pred = self.feedforward(x_batch1) 
                
                    self.backprop(y_batch1, y_pred, epoch)   
                       
            # After each epoch, it calculates the loss value for the validation data 
            loss_val = self.cost_func(self.xp, y_val, self.predict(x_val, status=0))
            
            # If the patience value is set, it checks if the loss value has improved
            if self.patience != 10000000000000:
                # If the loss has improved, it updates the best model and resets the waiting counter
                if loss_val < self._best_loss:
                    self._best_model, self._best_loss = self.layers, loss_val
                    self.waiting = 0
                # If the loss hasn't improved, it increments the waiting counter and checks if the patience limit has been reacher
                else: 
                    self.waiting +=1
                    print("not improving: [{}] current loss val: {} best: {}".format(self.waiting, loss_val, self._best_loss))
                    
                    # If the patience limit is reached, it reverts to the best model and stops training
                    if self.waiting >= self.patience:
                        self.layers = self._best_model
                        print("early stopping at epoch ", epoch)
                        return
            # If the epoch number is divisible by the verbose value, it calculates the loss value for 
            # the training data and updates the training history
            if epoch % verbose == 0:
                 loss_train = self.cost_func(self.xp, y_train, self.predict(x_train, status=0))
                 self._history['epochs'].append(epoch)
                 self._history['loss'].append(loss_train)
                 self._history['loss_val'].append(loss_val)
                 print("epoch: {0:=4}/{1} loss_train: {2:.8f} loss_val: {3:.8f} ".format(epoch, epochs, loss_train, loss_val))
                 

        # It returns the training history        
        self._history  
                
    def predict(self, x, status=1): 
        """
        Predicts the output for the given input data.

        Parameters:
        -----------
        x : array-like
            The input data for prediction.

        Returns:
        --------
        array-like
            The predicted output for the input data.
        """
        if status:
            input_dt = self.normalize_data(self.convert_data(x), self.mean_in, self.std_in)
            out = self.feedforward(input_dt)
            return self.denormalize_outputs(out, self.mean_out, self.std_out)
        else:
            out = self.feedforward(self.convert_data(x))
            return out
    
    def accuracy(self, y, y_pred):
        """
        Computes the accuracy of the predictions.
    
        Parameters:
        -----------
        y : array-like
            The true labels or target values.
        y_pred : array-like
            The predicted values.
    
        Returns:
        --------
        float
            The accuracy of the predictions as a percentage.
        """
        if isinstance(y, type(y_pred)) and isinstance(y_pred, type(y)):
            return 100*(1-self.xp.mean(self.xp.abs(y-y_pred)))
        else:
            print("Datas have different types.")
            return 0
    def addLayer(self): 
        pass
    
    def update_learning_rate(self, epoch):
        """
        Updates the learning rates of all layers based on the current epoch.
    
        Parameters:
        -----------
        epoch : int
            The current epoch number.
        """
        for layer in self.layers:
            for i in range(len(self.layers)):  # Itera sobre os 4 tipos de learning rate
                layer.learning_rates[i] = layer.lr_decay_method(layer.learning_rates[i], epoch, layer.lr_decay_rate, layer.lr_decay_steps)
            
    def _get_optimizer(self, optimizer_class):
        """
        Creates an instance of the specified optimizer class.
    
        Parameters:
        -----------
        optimizer_class : class
            The class of the optimizer to be instantiated.
    
        Returns:
        --------
        instance
            An instance of the specified optimizer class.
        """
        return optimizer_class()

    def verification(self, dt1):
        """
        Verifies the input data type for optimal performance of the RosenPY framework.
    
        Parameters:
        -----------
        dt1 : array-like
            The input data.
    
        """     
        if not isinstance(dt1, self.xp.ndarray):
              print("For optimal performance of the RosenPY framework, when not using GPU, input the data in NUMPY format, "
              "and when utilizing GPU, input the data in CUPY format.\n\n")
              
    def convert_data(self, data):
        """
        Converts the input data to the appropriate format for the current backend (NUMPY or CUPY).
    
        Parameters:
        -----------
        data : array-like
            The input data.
    
        Returns:
        --------
        array-like
            The converted input data.
        """
        
        if isinstance(data, self.xp.ndarray):
            return data
        if self.xp.__name__ == "cupy":
            return self.xp.asarray(data)
        if self.xp.__name__ == "numpy":
            return data.get()
        raise ValueError("Unsupported data type")
    
    def getHistory(self):
        """
        Returns the training history of the neural network.
    
        Returns:
        --------
        dict
            A dictionary containing the training history.
        """
        return self._history
    
    def normalize_data(self, input_data, mean=0, std_dev=0):
        # Implement the generic normalization logic
        return input_data

    def denormalize_outputs(self, normalized_output_data, mean=0, std_dev=0):
        # Implement the generic denormalization logic
        return normalized_output_data
    
    
    