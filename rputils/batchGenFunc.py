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

This file contains the functions used to produce batches. The batches will be 
sequential or shuffled.

"""

def batch_sequential(xp, x, y, batch_size=1):
    """
    Generates sequential batches of data for neural network training.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x: array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where n_samples is the number of
        samples and n_inputs is the number of input features.
        
    y: array-like, shape (n_samples, n_outputs)
        Target values are real numbers representing the desired outputs.
    
    batch_size : int, optional
        Batch sizes. If the batch size is equal to 1, the algorithm will 
        be the SGD.The default is 1.

    Returns
    -------
    array-like (n_batch, n_samples, n_inputs), array-like (n_batch, n_samples, n_outputs)
            Tensors that yields sequential batches of data.

    """
    n_batches = (x.shape[0] + batch_size - 1) // batch_size
    x_batch, y_batch = [], []
    for i in range(n_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        x_batch.append(x[start:end])
        y_batch.append(y[start:end])
    return xp.array(x_batch), xp.array(y_batch)

def batch_shuffle(xp, x, y, batch_size=1):
    """
    Generates shuffled batches of data for neural network training.

    Parameters
    ----------
    xp: str        
        CuPy/Numpy module. This parameter is set at the time of 
        initialization of the NeuralNetwork class.
    x: array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where n_samples is the number of
        samples and n_inputs is the number of input features.
        
    y: array-like, shape (n_samples, n_outputs)
        Target values are real numbers representing the desired outputs.
    
    batch_size : int, optional
        Batch sizes. If the batch size is equal to 1, the algorithm will 
        be the SGD.The default is 1.

    Returns
    -------
    array-like (n_batch, n_samples, n_inputs), array-like (n_batch, n_samples, n_outputs)
            Tensors that yields shuffled batches of data.
    """
    shuffle_index = xp.random.permutation(range(x.shape[0]))
    return batch_sequential(x[shuffle_index], y[shuffle_index], batch_size)
