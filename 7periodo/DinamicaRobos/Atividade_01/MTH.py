#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:30:16 2020

@author: bernardo
"""

import numpy as np
from numpy.linalg import inv
from math import cos, sin, pi


def Rx(th):
    #th = th*pi/180
    R = np.mat( [ [ 1,      0,         0],
                  [ 0, cos(th), -sin(th)],
                  [ 0, sin(th),  cos(th)] ])
    return R

def Ry(th):
    #th = th*pi/180
    R = np.mat( [ [ cos(th), 0, sin(th)],
                  [       0, 1,       0],
                  [-sin(th), 0, cos(th)] ])
    return R

def Rz(th):
    #th = th*pi/180
    R = np.mat( [ [ cos(th), -sin(th), 0],
                  [ sin(th),  cos(th), 0],
                  [       0,        0, 1] ])
    return R

def H(R,d):
    _H = np.block([ [R, d], [0,0,0,1] ])
    return _H

d_10 = np.mat( [ [1.34], [0.00], [1.22] ] )
d_20 = np.mat( [ [1.00], [0.30], [0.60] ] )

H_10 = H( Ry(120), d_10)
H_20 = H( Ry(135), d_20)

H_21 = inv(H_10)*H_20
print('\t H de 2 para 1:\n')
print(H_21)





