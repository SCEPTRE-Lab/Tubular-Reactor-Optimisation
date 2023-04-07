from sciann_datagenerator import DataGeneratorXYT

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
#tf.disable_v2_behavior()
from numpy.random import seed
import random


Length_start = 0
Length_end = 10

# Controlling the randomness
_seed = 1256
seed(_seed)
random.seed(_seed)
tf.random.set_seed(_seed)
tf.compat.v1.set_random_seed(_seed)


dg = DataGeneratorXYT(X=[0.,10.], Y=[0.,1.], T = [300,400],
                     num_sample=3000,
                     targets=4*['domain'] + 3*['bc-left'] + 3*['bc-bot'] + 3*['bc-top'])

#dg.plot_data()
input_data, target_data = dg.get_data()


# Solving TUBULAR REACTORS WITH LAMINAR FLOW


ubar = 0.5
D = 0.1
z = sn.Variable('z')
z_s = (z-0)/10

r = sn.Variable('r')
r_s = (r+0)/1

T = sn.Variable('T')
T_s = (T-300)/100



Ca,Cb,Cc= sn.Functional(['Ca','Cb','Cc'], [z_s,r_s,T_s],4*[32], 'tanh')

from numpy import pi
from sciann.utils.math import diff, sign, sin

#kt=0.1
#r=0
U = 2*ubar*(1-np.square(r))

R = 8.314

K1 = 22000
E1 = 43000

K2 = 90000
E2 = 45000

k = K1*sn.exp(-E1/(R*T))
k2 = K2*sn.exp(-E2/(R*T))

r1 = k*sn.pow(Ca,2)
r2 = -k*sn.pow(Ca,2)+k2*Cb
r3 = -k2*Cb
L1 = r*D*sn.diff(Ca,z,order=2)-U*r*sn.diff(Ca, z) + D * r *sn.diff(Ca,r,order = 2) + D * sn.diff(Ca,r) - r1*r
L2 = r*D*sn.diff(Cb,z,order=2)-U*r*sn.diff(Cb, z) + D * r *sn.diff(Cb,r,order = 2) + D * sn.diff(Cb,r) - r2*r
L3 = r*D*sn.diff(Cc,z,order=2)-U*r*sn.diff(Cc, z) + D * r *sn.diff(Cc,r,order = 2) + D * sn.diff(Cc,r) - r3*r
L4 = 100 - (Ca+Cb+Cc)
#L3 = r*D*sn.diff(Cc,z,order=2)-U*r*sn.diff(Cc, z) + D * r *sn.diff(Cc,r,order = 2) + D * sn.diff(Cc,r) - k*Cc*r

#L1 = r*(1-sn.math.square(r))*sn.diff(C, z) - (0.1*(sn.diff(C,r) + r*sn.diff(C,r,order = 2)) - C*r)
C1 = (z == 0) * (Ca-100)
C1_2 = (z == 0) * (Cb-0.0)
C1_3 = (z == 0) * (Cc-0.0)
# C2 = (r == 0) * (diff(Ca,r))
# C3 = (r == 1) * (diff(Ca,r))

C2 = (r == 0) * (diff(Ca,r))
C2_2 = (r == 0) * (diff(Cb,r))
C2_3 = (r == 0) * (diff(Cc,r))


C3 = (r == 1) * (diff(Ca,r))
C3_2 = (r == 1) * (diff(Cb,r))
C3_3 = (r == 1) * (diff(Cc,r))
m = sn.SciModel([z,r,T], [L1,L2,L3,L4, C1,C1_2,C1_3, C2,C2_2,C2_3, C3,C3_2,C3_3])
#m.load_weights('SciANN_FixedTemp_50k_4_32.hdf5')
# #load_weights_from
a = 100
b = 30
z_test, r_test = np.meshgrid(
    np.linspace(0, 10, a),
    np.linspace(0, 1, b)
)

# #
T_test = np.ones(a*b).reshape([b,a])*300

#
input_data = [z_test, r_test, T_test]

Ca_pred = Ca.eval(m,input_data)
Cb_pred = Cb.eval(m,input_data)
Cc_pred = Cc.eval(m,input_data)
#
# fig5 = plt.figure(figsize=(8,5))
# plt.plot(z_test[39,:],Ca_pred[0,:],label='Ca at R=0')
# plt.plot(z_test[39,:],Cb_pred[0,:],label='Cb at R=0')
# plt.plot(z_test[39,:],Cc_pred[0,:],label='Cc at R=0')
# plt.xlabel('Z')
# plt.ylabel('R')



def optim_function(TP):
    TP = TP[0]
    # print("Value of A:",A)
    # print("Value of B:",B)

    A_test = np.ones(a * b).reshape([b, a]) * TP

    input_data = [z_test, r_test, A_test]

    Ca_pred = Ca.eval(m, input_data)
    Cb_pred = Cb.eval(m, input_data)
    Cc_pred = Cc.eval(m, input_data)

    Ca_sum = np.sum(Ca_pred[0, :])
    Cb_sum = np.sum(Cb_pred[0, :])
    Cc_sum = np.sum(Cc_pred[0, :])

    # print("Sum of A:",Ca_sum)
    # print("Sum of B:",Cb_sum)
    # print("Sum of C:",Cc_sum)
    # obj_value = -Cb_sum
    obj_value = Cc_pred[0, 99]
    # print(obj_value)
    print( TP, obj_value)

    return obj_value


print('{0:4s}   {1:9s}   {2:9s}   {3:9s}  '.format('Iter', ' X1', ' X2', 'f(X)'))

#optim_function(x0)

from scipy.optimize import Bounds, minimize

bounds = [(300,400)]
#optim_function(np.array([350]))
#x0 = 350
start = time.time()
res = minimize(optim_function, np.array([400]), method='powell',bounds = bounds)
#res = minimize(optim_function, x0, method='powell',options = {'xtol': 0.1, 'ftol': 0.00000001},bounds=bounds)

end = time.time()

print(f"Runtime of the program is {end - start}")
print(res.x)
# from scipy.optimize import rosen, differential_evolution
# bounds = Bounds(300,400)
# bounds = [(300,400)]
# #result = differential_evolution(optim_function, bounds = (300,400))
# result = differential_evolution(optim_function, bounds)
