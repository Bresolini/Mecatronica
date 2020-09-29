#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:36:14 2020

@author: bernardo

Codigo usado para computar a matriz de transformaçao homogenea de n para 0.
E preciso ser inserido a tabela DH na ordem:
            alf,   a,    d,   theta
"""

import sympy as sp    # Pacote para manipulaçoes simbolicas
from sympy import pi, cos, sin
from sympy.physics.vector import init_vprinting   # Usada para printar em latex
from sympy.physics.mechanics import dynamicsymbols

init_vprinting(use_latex='mathjax', pretty_print=False)   # Configurando o vprinting

theta1, theta2, theta3, theta4 = dynamicsymbols('theta1 theta2 theta3 theta4')
theta5, theta6, theta7 = dynamicsymbols('theta5 theta6 theta7')
l1, l2, theta, alpha, a, d = dynamicsymbols('l1 l2 theta alpha a d')

# Valores das distancias
dbs, dse, dew, dwf = 360, 420, 400, 90

# Matriz DH:      ai,  alfi,  di, theta
TDH = sp.Matrix([ [0, -pi/2, dbs, theta1],
                  [0,  pi/2,   0, theta2],
                  [0,  pi/2, dse, theta3],
                  [0, -pi/2,   0, theta4],
                  [0, -pi/2, dew, theta5],
                  [0,  pi/2,   0, theta6],
                  [0,     0, dwf, theta7]])

Rot = sp.Matrix([[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha)],
                 [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha)],
                 [         0,             sin(alpha),             cos(alpha)]])

Tran = sp.Matrix([a*cos(theta),a*sin(theta),d])
S = sp.Matrix([[0, 0, 0, 1]])

# Matriz
A = sp.Matrix.vstack(sp.Matrix.hstack(Rot, Tran), S)


# Matriz A de cada linha
A1 = A.subs({ alpha:TDH[0,0], a:TDH[0,1], d:TDH[0,2], theta:TDH[0,3] })
A2 = A.subs({ alpha:TDH[1,0], a:TDH[1,1], d:TDH[1,2], theta:TDH[1,3] })
A3 = A.subs({ alpha:TDH[2,0], a:TDH[2,1], d:TDH[2,2], theta:TDH[2,3] })
A4 = A.subs({ alpha:TDH[3,0], a:TDH[3,1], d:TDH[3,2], theta:TDH[3,3] })
A5 = A.subs({ alpha:TDH[4,0], a:TDH[4,1], d:TDH[4,2], theta:TDH[4,3] })
A6 = A.subs({ alpha:TDH[5,0], a:TDH[5,1], d:TDH[5,2], theta:TDH[5,3] })
A7 = A.subs({ alpha:TDH[6,0], a:TDH[6,1], d:TDH[6,2], theta:TDH[6,3] })

# Matriz de transf. homogenea de 7 para 0
T = A1*A2*A3*A4*A5*A6*A7
Thome = T.subs({ theta1:0, theta2:0, theta3:0, theta4:0, theta5:0, theta6:0, theta7:0 })

q1, q2, q3, q4, q5, q6, q7 = 0, 0, 0, -pi/2, pi/3, 0, 0

T_ = T.subs({ theta1:q1, theta2:q2, theta3:q3, theta4:q4, theta5:q5, theta6:q6, theta7:q7 })

"""
T = sp.Matrix([ [T[0,0].simplify(), T[0,1].simplify(), T[0,2].simplify(), sp.trigsimp(T[0,3].simplify())],
                [T[1,0].simplify(), T[1,1].simplify(), T[1,2].simplify(), sp.trigsimp(T[1,3].simplify())],
                [T[2,0].simplify(), T[2,1].simplify(), T[2,2].simplify(), sp.trigsimp(T[2,3].simplify())],
                [0, 0, 0, 1] ])



T_ = sp.Matrix([[T[0,0].simplify(), T[0,1].simplify(), sp.trigsimp(T[0,3].simplify())],
                 [T[1,0].simplify(), T[1,1].simplify(), sp.trigsimp(T[1,3].simplify())],
                 [T[2,0].simplify(), T[2,1].simplify(), T[2,2].simplify()]])

"""



















