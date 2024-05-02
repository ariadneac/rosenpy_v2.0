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

import rosenpymodel.ptrbfnnc as mynn
import rputils.utils as utils
import rputils.initFunc as initFunc
import rputils.decayFunc as decayFunc
import rosenpymodel.rpoptimizer as opt
import dataset.beamforming as dt
import numpy as np

def setData():
    """
    Set up the data for training.

    Returns:
        tuple: Tuple containing the normalized input and output datasets.
    """
    f = 850e6
    SINRdB = 20
    SNRdBs = 25
    SNRdBi = 20
    phi = [1, 60, 90, 120, 160, 200, 240, 260, 280, 300, 330]
    theta = [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
    desired = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    modulation = ["QAM", "WGN", "QAM", "PSK", "QAM", "WGN", "QAM", "WGN", "QAM", "PSK", "PSK"]
    Mmod = [4, 0, 64, 8, 256, 0, 16, 0, 64, 16, 8]

    lenData = int(1e4)

    # Converting 'desired' to a NumPy array
    desired = np.array(desired)

    
    # Calling the create_dataset_beam function
    SetIn, SetOut = dt.create_dataset_beam(modulation, Mmod, f, phi, theta, desired, lenData, SINRdB, SNRdBs, SNRdBi)
    

    return SetIn, SetOut

###############################################################################################################
###############################################################################################################


# Initialize input_data and output_data using the setData function
input_data, output_data = setData()

# Create an instance of the PTRBF Neural Network
nn = mynn.PTRBFNNConv(gpu_enable=False)

# Add layers to the network
nn.addLayer(layer_type="Conv", ishape=input_data.shape[1], neurons=10,
            weights_initializer=initFunc.opt_ptrbf_weights,
            bias_initializer=initFunc.zeros,
            gamma_initializer=initFunc.opt_ptrbf_gamma,
            sigma_initializer=initFunc.ones,
            lr_decay_method=decayFunc.time_based_decay,
            gamma_rate=1e-2,
            sigma_rate=1e-2)

nn.addLayer(layer_type="Fully", neurons=10, oshape=output_data.shape[1],
            weights_initializer=initFunc.opt_ptrbf_weights,
            bias_initializer=initFunc.zeros,
            gamma_initializer=initFunc.opt_ptrbf_gamma,
            sigma_initializer=initFunc.ones,
            gamma_rate=1e-2,
            sigma_rate=1e-2)

# Train the neural network using fit method
nn.fit(input_data, output_data, epochs=500, verbose=100, batch_size=100, optimizer=opt.CVAdam())

# Make predictions using the trained model
y_pred = nn.predict(input_data)

# Calculate and print accuracy
print('Accuracy: {:.2f}%'.format(nn.accuracy(output_data, y_pred)))

