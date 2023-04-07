from sciann_datagenerator import DataGeneratorXYT

import numpy as np
import matplotlib.pyplot as plt
import sciann as sn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dg = DataGeneratorXYT(X=[0.,10.], Y=[0.,1.], T = [300,400],
                     num_sample=3000,
                     targets=4*['domain'] + 3*['bc-left'] + 1*['bc-bot'] + 1*['bc-top'])

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



Ca,Cb,Cc= sn.Functional(['Ca','Cb','Cc'], [z_s,r_s,T_s], 4*[64], 'tanh')

from numpy import pi
from sciann.utils.math import diff, sign, sin

#kt=0.1
#r=0
U = 2*ubar*(1-np.square(r))
K0 = 10000
E = 40230
R = 8.314


k = K0*sn.exp(-E/(R*T))
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
C3 = (r == 0) * (diff(Ca,r))
C4 = (r == 1) * (diff(Ca,r))
m = sn.SciModel([z,r,T], [L1,L2,L3,L4, C1,C1_2,C1_3,C3, C4])#, optimizer='ftrl')


# z_data, r_data = np.meshgrid(
#     np.linspace(0, 10, 100),
#     np.linspace(-1, 1, 100)
# )
h = m.train(input_data, target_data, learning_rate=0.001, epochs=30000, verbose=1,stop_loss_value=1e-9)
#,adaptive_weights = {"method": "NTK", "freq": 100},

a = 200
z_test, r_test= np.meshgrid(
    np.linspace(0, 10, a),
    np.linspace(0, 1, a)
)
T_test = np.ones(a**2).reshape([a,a])*350

#
input_data = [z_test, r_test, T_test]
C_pred = Cc.eval(m, input_data)
Ca_pred = Ca.eval(m, input_data)
Cb_pred = Cb.eval(m, input_data)
Cc_pred = Cc.eval(m, input_data)

import pandas as pd

pd.DataFrame(Ca_pred).to_clipboard()
pd.DataFrame(Cb_pred).to_clipboard()
pd.DataFrame(Cc_pred).to_clipboard()

#SUBPLOTS

plt.rcParams["figure.figsize"] = [12, 9]
plt.rcParams["figure.autolayout"] = True
fig, axs = plt.subplots(3)
#plt.suptitle("Concentration of Species at T = 320k")

ax = axs[0]
pcm = ax.pcolor(z_test,r_test,Ca_pred, cmap='seismic',shading='auto')
ax.set_ylabel('Radius of Reactor (r)',fontsize = 14.0)
ax.set_title('Concentration of A')
ax.tick_params(axis='both', which='major', labelsize=12)
fig.colorbar(pcm, ax=ax)

ax = axs[1]
pcm = ax.pcolor(z_test,r_test,Cb_pred, cmap='seismic',shading='auto')
ax.set_ylabel('Radius of Reactor (r)',fontsize = 14.0)
ax.set_title('Concentration of B')

ax.tick_params(axis='both', which='major', labelsize=12)
fig.colorbar(pcm, ax=ax)

ax = axs[2]
pcm = ax.pcolor(z_test,r_test,Cc_pred, cmap='seismic',shading='auto')
ax.set_ylabel('Radius of Reactor (r)',fontsize = 14.0)
ax.set_title('Concentration of C')

ax.set_xlabel('Length along the reactor (z)',fontsize = 14.0)
ax.tick_params(axis='both', which='major', labelsize=12)
fig.colorbar(pcm, ax=ax)
plt.show()




fig5 = plt.figure(figsize=(8,4))
plt.plot(z_test[0,:],Ca_pred[0,:],label='Ca at R=0')
plt.plot(z_test[0,:],Cb_pred[0,:],label='Cb at R=0')
plt.plot(z_test[0,:],Cc_pred[0,:],label='Cc at R=0')
#
plt.xlabel('Z')
plt.ylabel('C')
plt.title('Concentration along length of reactor at T=350')
plt.legend()

pd.DataFrame(z_test[0,:]).to_clipboard()
pd.DataFrame(Ca_pred[0,:]).to_clipboard()
pd.DataFrame(Cb_pred[0,:]).to_clipboard()
pd.DataFrame(Cc_pred[0,:]).to_clipboard()

#pd.concat([pd.DataFrame(z_test[0,:]),pd.DataFrame(Ca_pred[0,:]), pd.DataFrame(Cb_pred[0,:]),pd.DataFrame(Cc_pred[0,:])], axis = 1).to_clipboard()


# #
# m.save_weights('ABC_200000epochs_300_400_Z_R_T.hdf5')
# #
# fig =  plt.figure()
# loss_val = h.history['loss']#/h.history['loss'][0]
# plt.semilogy(loss_val)
# plt.xlabel('epochs')
# plt.ylabel('$\\mathcal{L}/\\mathcal{L}_0$')
#
# plt.show()
