#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:21:12 2019

@author: agniv
"""

import numpy as np
import math


def dot_product(a, b):
    return (a[0][0] * b[0][0]) + (a[1][0] * b[1][0]) + (a[2][0] * b[2][0])


p1 = np.array([[0.0], [2.0], [0.0]])

p2 = np.array([[0.0], [0.0], [2.0]])

p3 = np.array([[0.0], [1.0], [0.0]])

Ps = np.array([[-10.0], [0.0], [0.0]])


# the un-normalized vector from point p3 to p1
p3_p1 = np.array([[p3[0][0] - p1[0][0]], [p3[1][0] - p1[1][0]], [p3[2][0] - p1[2][0]]])

# the un-normalized vector from point p2 to p1
p2_p1 = np.array([[p2[0][0] - p1[0][0]], [p2[1][0] - p1[1][0]], [p2[2][0] - p1[2][0]]])

print(p2_p1)
print(p3_p1)

a1 = p3_p1[0][0]
a2 = p3_p1[1][0]
a3 = p3_p1[2][0]
b1 = p2_p1[0][0]
b2 = p2_p1[1][0]
b3 = p2_p1[2][0]

# the un-normalized normal to the triangular plane
n_ = np.array(
    [[(a2 * b3) - (a3 * b2)], [(a3 * b1) - (a1 * b3)], [(a1 * b2) - (a2 * b1)]]
)

mag_n_ = math.sqrt(
    (n_[0][0] * n_[0][0]) + (n_[1][0] * n_[1][0]) + (n_[2][0] * n_[2][0])
)

# unit normal to the plane
n = n_ / mag_n_

print("Plane normal:")
print(n_)

print("K prio")
print(np.matmul(n, np.transpose(n)))
# Matrix K
K = (np.identity(3) - np.matmul(n, np.transpose(n))) / mag_n_
# K = np.identity(3)


x1 = p1[0][0]
y1 = p1[1][0]
z1 = p1[2][0]
x2 = p2[0][0]
y2 = p2[1][0]
z2 = p2[2][0]
x3 = p3[0][0]
y3 = p3[1][0]
z3 = p3[2][0]


# un-normalized vector from p1 to Ps
n_Ps = np.array([[Ps[0][0] - p1[0][0]], [Ps[1][0] - p1[1][0]], [Ps[2][0] - p1[2][0]]])


# the elements of J_elem_1

t1_1 = ((y3 - y2) * n[2][0]) - ((z3 - z2) * n[1][0])
J1_1_a = np.array([[t1_1 * n[0][0]], [t1_1 * n[1][0]], [t1_1 * n[2][0]]])

print("1st mult:")
print(np.matmul(K, J1_1_a))

J1_1 = -n[0][0] + dot_product(n_Ps, np.matmul(K, J1_1_a))


t1_2 = ((z3 - z2) * n[0][0]) - ((x3 - x2) * n[2][0])
J1_2_a = np.array([[t1_2 * n[0][0]], [t1_2 * n[1][0]], [t1_2 * n[2][0]]])
J1_2 = -n[1][0] + dot_product(n_Ps, np.matmul(K, J1_2_a))

print("2nd mult:")
print(np.matmul(K, J1_2_a))


t1_3 = ((x3 - x2) * n[1][0]) - ((y3 - x2) * n[1][0])
J1_3_a = np.array([[t1_3 * n[0][0]], [t1_3 * n[1][0]], [t1_3 * n[2][0]]])
J1_3 = -n[2][0] + dot_product(n_Ps, np.matmul(K, J1_3_a))

print("3rd mult:")
print(np.matmul(K, J1_3_a))


J1_4_a = -np.array([[0.0], [z1 - z3], [y3 - y1]])
J1_4 = dot_product(n_Ps, np.matmul(K, J1_4_a))


J1_5_a = -np.array([[z3 - z1], [0.0], [x1 - x3]])
J1_5 = dot_product(n_Ps, np.matmul(K, J1_5_a))


J1_6_a = -np.array([[y1 - y3], [x3 - x1], [0.0]])
J1_6 = dot_product(n_Ps, np.matmul(K, J1_6_a))


J1_7_a = -np.array([[0.0], [z2 - z1], [y1 - y2]])
J1_7 = dot_product(n_Ps, np.matmul(K, J1_7_a))


J1_8_a = -np.array([[z1 - z2], [0.0], [x2 - x1]])
J1_8 = dot_product(n_Ps, np.matmul(K, J1_8_a))


J1_9_a = -np.array([[y2 - y1], [x1 - x2], [0.0]])
J1_9 = dot_product(n_Ps, np.matmul(K, J1_9_a))


print("J1_9:")
print(J1_9)
print("n_Ps:")
print(n_Ps)
print("np.matmul(K,J1_9_a):")
print(np.matmul(K, J1_9_a))

J1 = np.array([J1_1, J1_2, J1_3, J1_4, J1_5, J1_6, J1_7, J1_8, J1_9])

J2 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
J = np.matmul(J1, J2)

print("*****************")
print("JACOBIAN:")
print(J)
print("*****************")
print("JACOBIAN - 1st matrix:")
print(J1)
print("JACOBIAN - 2nd matrix:")
print(J2)
