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
"""

This file contains all the activation functions used by RosenPy.

"""
from rputils import gpu

#xp = gpu.module()
import numpy as xp

def sinh(module, x, derivative=False):
    """
    Activation function - Hyperbolic sine, element-wise.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.
   
    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    if derivative:
        return xp.cosh(x)
    return xp.sinh(x)

def atanh(module, x, derivative=False):
    """
    Activation function - the inverse hyperbolic tangent , element-wise.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    if derivative:
        return 1/(1-xp.square(x))
    return xp.arctanh(x)

def asinh(module, x, derivative=False):
    """
    Activation function - inverse hyperbolic sine , element-wise.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    if derivative:
        return 1/(1+xp.square(x))
    return xp.arcsinh(x)

def tan(module, x, derivative=False):
    """
    Activation function - tangent , element-wise.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    if derivative:
        return 1/(xp.square(xp.cos(x)))
    return xp.tan(x)

def sin(module, x, derivative=False):
    """
    Activation function - sine, element-wise.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    if derivative:
        return xp.cos(x)
    return xp.sin(x)

def atan(module, x, derivative=False):  
    """
    Activation function - arc tangent, element-wise.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    if derivative:
        return 1/(1+xp.square(x))
    return xp.arctan(x)

def asin(module, x, derivative=False):
    """
    Activation function - arc sine, element-wise.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    if derivative:
        return 1/xp.sqrt((1-xp.square(x)))
    return xp.arcsin(x)

def acos(module, x, derivative=False):
    """
    Activation function - arc cosine, element-wise.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    if derivative:
        return 1/xp.sqrt((xp.square(x)-1))
    return xp.arccos(x)

def sech(module, x, derivative=False):
    """
    Activation function - the hyperbolic secant, element-wise.
    This is the FCRBFNN activation function.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    
    ex = xp.exp(x)
    if derivative:
        return -2 * ex / (ex + 1 / ex) ** 2
    return 2 / (ex + 1 / ex)

def linear(module, x, derivative=False): 
    """
    The linear activation function, also known as "no activation," or 
    "identity function" (multiplied x1.0)

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x : array_like
        Input array.
    derivative : bool, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    return xp.ones_like(x) if derivative else x

def tanh(module, x, derivative=False):
     """
     Activation function - Hyperbolic tangent, element-wise.

     Parameters
     ----------
     xp: str        
         CuPy/Numpy module. This parameter is set at the time of 
         initialization of the NeuralNetwork class.
     x : array_like
         Input array.
     derivative : bool, optional
         The default is False.

     Returns
     -------
     array_like
         It defines whether what will be returned will be the activation 
         function (feedforward) or its derivative (backpropagation).

     """
     if derivative:
         return 1-xp.square(xp.tanh(x))   
     return xp.tanh(x)

def splitComplex(module, y, act, derivative=False):
    """
    This function is used in SCFFNN, since he activation functions that are separately
    applied to the real and imaginary components of the linear combination of
    each layer.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    y : array_like
        Input array.
    act : str
        Name of the activation function that will be used in the SCFFNN classes, as string.
    derivative : TYPE, optional
        The default is False.

    Returns
    -------
    array_like
        It defines whether what will be returned will be the activation 
        function (feedforward) or its derivative (backpropagation).

    """
    return act(module, xp.real(y), derivative) + 1.0j*act(module, xp.imag(y), derivative)