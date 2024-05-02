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

import numpy as np


def module(module):
    """
    This function is used to implement CPU/GPU generic code.  
    
    Returns
    -------
    module
        It returns module `cupy` or `numpy`.

    """
    if module:
        try:
            import cupy as xp
            print("Cupy module")
        except ImportError:
            print("Cupy is not installed. Using Numpy instead.")
            xp = np
    else: 
        print("Numpy module")
        xp = np
        
    return xp

