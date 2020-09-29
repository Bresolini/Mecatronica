#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:22:54 2020

@author: bernardo
"""
import numpy as np
from math import exp


"""
Considere a funçao
        $$ f(w) = w e^w - z = 0  $$
sendo z uma constante.

A aplicaçao do metodo de Halley¹ leva
 $$ w_{k+1} = w_k - \frac{ f }{ e^{w_k}(w_k+1) - \frac{(w_k+2)f}{2w_k + 2} } $$
cuja convergencia eh cubica.

¹ O metodo de Halley eh dado por
     $$ x_{k+1} = x_k - \frac{ f }{ f' - \frac{f \cdot f''}{ 2 f' } } $$
"""
# Metodo numerico para determinar a funçao W de Lambert para um z dado
#                  w*exp(w) = z
def lambert(w,z):
    f = w*exp(w) - z
    return f
z  = -0.02/np.e

w = np.empty( (5,) )
w.fill(np.nan)

w[0] = -6
w[1] = w[0] - lambert(w[0],z)/ \
( exp(w[0])*(w[0]+1) - (w[0]+2)*lambert(w[0],z)/(2*w[0]+2)  )

for i in range(1,len(w)-1):
    w[i+1] = w[i] - lambert(w[i],z)/ \
    ( exp(w[i])*(w[i]+1) - (w[i]+2)*lambert(w[i],z)/(2*w[i]+2)  )

print(w)
