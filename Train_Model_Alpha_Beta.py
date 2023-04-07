from sciann_datagenerator import DataGeneratorXY
import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, sin
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

Length_start = 0
Length_end = 10

dg = DataGeneratorXY(X=[Length_start,Length_end], Y=[-1.,1.],
                     num_sample=3000,
                     targets=4*['domain'] + 3*['bc-left'] + 1*['bc-bot'] + 1*['bc-top']+ 1*['cen'])

#dg.plot_data()
input_data, target_data = dg.get_data()

alpha = np.random.uniform(2,20, 4125).reshape(4125,1)
beta= np.random.uniform(2,20, 4125).reshape(4125,1)
input_data.append(alpha)
input_data.append(beta)

# Solving TUBULAR REACTORS WITH LAMINAR FLOW

ubar = 0.5
D = 0.1
z = sn.Variable('z')
z_s = (z-0)/10

r = sn.Variable('r')
r_s = (r+1)/2

A = sn.Variable('A')
A_s = (A-2)/18

B = sn.Variable('B')
B_s = (B-2)/18


Ca,Cb,Cc= sn.Functional(['Ca','Cb','Cc'], [z_s,r_s,A_s,B_s], 8*[32], 'tanh')

U = 2*ubar*(1-np.square(r))
K0 = 10000
E = 40230
R = 8.314


T0 = 330
epss = 0.000
#z = np.linspace(0,10,100)
beta_func = (np.log(2.50662827463)+(A-0.5)*sn.log(A)+(B-0.5)*sn.log(B) - (A+B-0.5)*sn.log(A+B))
beta_dist = (A-1)*sn.log(z+epss)+(B-1)*sn.log(10-z+epss)-(beta_func+(A+B-1)*np.log(10))
TP = T0 + 100*sn.exp(beta_dist)

#TP=4.8*z+0.2*sn.pow(z,2)+T

k = K0*sn.exp(-E/(R*TP))
k2 = 1*k
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

C2 = (r == -1) * (diff(Ca,r))
C3 = (r == 1) * (diff(Ca,r))
C4 = (r == 0) * (diff(Ca,r))
m = sn.SciModel([z,r,A,B], [L1,L2,L3,L4, C1,C1_2,C1_3, C2, C3, C4])#, optimizer='ftrl')


# z_data, r_data = np.meshgrid(
#     np.linspace(0, 10, 100),
#     np.linspace(-1, 1, 100)
# )
h = m.train(input_data, target_data, learning_rate=0.001, epochs=30000, verbose=1,stop_loss_value=1e-8)
#,adaptive_weights = {"method": "NTK", "freq": 100},
a = 200
z_test, r_test= np.meshgrid(
    np.linspace(0, 10, a),
    np.linspace(-1, 1, a)
)
A_test = np.ones(a**2).reshape([a,a])*2
B_test = np.ones(a**2).reshape([a,a])*20
#
input_data = [z_test, r_test, A_test,B_test]
C_pred = Cc.eval(m, input_data)
Ca_pred = Ca.eval(m, input_data)
Cb_pred = Cb.eval(m, input_data)
Cc_pred = Cc.eval(m, input_data)
#
#
# #
# #
# # 2D Heatmap
fig = plt.figure(figsize=(8,4))
plt.pcolor(z_test,r_test,Ca_pred, cmap='seismic',shading='auto')
plt.xlabel('Z')
plt.ylabel('R')
plt.title('A=0.3 ,B=1 ,C=350 for Ca')
plt.colorbar()

fig2 = plt.figure(figsize=(8,4))
plt.pcolor(z_test,r_test,Cb_pred, cmap='seismic',shading='auto')
plt.xlabel('Z')
plt.ylabel('R')
plt.title('A=0.3 ,B=1 ,C=350 for Cb')
plt.colorbar()

fig3 = plt.figure(figsize=(8,4))
plt.pcolor(z_test,r_test,Cc_pred, cmap='seismic',shading='auto')
plt.xlabel('Z')
plt.ylabel('R')
plt.title('A=0.3 ,B=1 ,C=350 for Cc')
plt.colorbar()
# #

fig5 = plt.figure()
plt.plot(z_test[100,:],Ca_pred[100,:],label='Ca at R=0')
plt.plot(z_test[100,:],Cb_pred[100,:],label='Cb at R=0')
plt.plot(z_test[100,:],Cc_pred[100,:],label='Cc at R=0')
#
# plt.xlabel('Z')
# plt.ylabel('C')
# plt.title('C vs Z at A=0.3 ,B=1 ,C=350')
# plt.legend()
#
#m.save_weights('optim_z_r_T_50000_mb.hdf5')

fig =  plt.figure()
loss_val = h.history['loss']#/h.history['loss'][0]
plt.semilogy(loss_val)
plt.xlabel('epochs')
plt.ylabel('$\\mathcal{L}/\\mathcal{L}_0$')

plt.show()
