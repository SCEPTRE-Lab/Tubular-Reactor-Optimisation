import pandas as pd

from sciann_datagenerator import DataGeneratorXY
import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin
import time


ubar = 0.5
D = 0.1
z = sn.Variable('z')
z_s = (z-0)/10

r = sn.Variable('r')
r_s = (r+0)/1

A = sn.Variable('A')
A_s = (A-2)/18

B = sn.Variable('B')
B_s = (B-2)/18


Ca,Cb,Cc = sn.Functional(['Ca','Cb','Cc'], [z_s,r_s,A_s,B_s], 5*[64], activation='tanh')
# Ca = Ca_s*100 + 0
# Cb = Cb_s*100 + 0
# Cc = Cc_s*100 + 0
U = 2*ubar*(1-np.square(r))

K1 = 22000
E1 = 43000

K2 = 90000
E2 = 45000
R = 8.314


T0 = 290
epss = 0.000
#z = np.linspace(0,10,100)
beta_func = (np.log(2.50662827463)+(A-0.5)*sn.log(A)+(B-0.5)*sn.log(B) - (A+B-0.5)*sn.log(A+B))
beta_dist = (A-1)*sn.log(z+epss)+(B-1)*sn.log(10-z+epss)-(beta_func+(A+B-1)*np.log(10))
TP = T0 + 100*sn.exp(beta_dist)

k = K1*sn.exp(-E1/(R*TP))
k2 = K2*sn.exp(-E2/(R*TP))
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

#m = sn.SciModel([z,r,A,B], [L1,L2,L3,L4, C1,C1_2,C1_3, C2, C3])#, optimizer='ftrl')
m = sn.SciModel([z,r,A,B], [L1,L2,L3,L4, C1,C1_2,C1_3, C2,C2_2,C2_3, C3,C3_2,C3_3])#, optimizer='ftrl')
#m.load_weights('SciANN_ALPHA_BETA_2_20_290_30K_EPOCHS_Piecewise_Params_5_64.hdf5')
# #load_weights_from
a = 100
b = 40
z_test, r_test = np.meshgrid(
    np.linspace(0, 10, a),
    np.linspace(0, 1, b)
)

# #
A_test = np.ones(a*b).reshape([b,a])*2
B_test = np.ones(a*b).reshape([b,a])*20

#
input_data = [z_test, r_test, A_test,B_test]

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



def optim_function(x,info):
    AA = x[0]
    BB = x[1]
    # print("Value of A:",A)
    # print("Value of B:",B)

    A_test = np.ones(a * b).reshape([b, a]) * AA
    B_test = np.ones(a * b).reshape([b, a]) * BB
    input_data = [z_test, r_test, A_test, B_test]

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
    print(info['Nfeval'], AA, BB, obj_value)
    info['Nfeval'] += 1

    return obj_value


print('{0:4s}   {1:9s}   {2:9s}   {3:9s}  '.format('Iter', ' X1', ' X2', 'f(X)'))

#optim_function(x0)

from scipy.optimize import Bounds, minimize

bounds = Bounds([2.0, 2.0], [20.0, 20.0])

# x0 = np.array([10,10])
# res = minimize(optim_function, x0, method='Nelder-Mead', bounds=bounds)
#res = minimize(optim_function, x0, method='powell',options = {'xtol': 0.1, 'ftol': 0.00000001},bounds=bounds)

from scipy.optimize import rosen, differential_evolution, shgo
#bounds = Bounds([2,2], [20, 20])
bounds = [(2,20), (2, 20)]
start = time.time()
result = shgo(optim_function, args=({'Nfeval': 0},), bounds=bounds )
#result = differential_evolution(optim_function, bounds, maxiter=100, disp = True)
print(result.x)
print(result)
end = time.time()

print(f"Runtime of the program is {end - start}")
