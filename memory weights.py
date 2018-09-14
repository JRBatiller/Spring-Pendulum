# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:10:52 2018

@author: Joby
"""



from scipy.integrate import odeint
from scipy.fftpack import fft, fftfreq, fftshift
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import sklearn.cross_validation
import matplotlib as mpl
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

pi=np.pi

def get_output(directory):
    dataDir = "C:/Users/Joby/Documents/MSc Summer Project/"+directory+"/"
    mats = []
    file_name_list=os.listdir( dataDir )
    for file in file_name_list :
        mats.append( sio.loadmat( dataDir+file ) )
    
    return mats, file_name_list

def normalize_coefs(coefs):
    #should be a list
    normed_coefs=[]
    for coef in coefs:
        coef = abs(coef)
        coef = coef/max(coef)
        normed_coefs.append(coef)
    return normed_coefs


def compartment_coef(coefs):
    comp_coefs=[]
    for coef in coefs:
        ccoef=coef.reshape((len(coef)/5,5))
        ccoef=np.mean(ccoef,1)
        comp_coefs.append(ccoef)
    return comp_coefs

#plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'font.size': 18})

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 4
plt.rcParams["figure.figsize"] = fig_size



contents=sio.loadmat('memory_error_10.0_1.0.mat')
coef_list = contents['coef_list']
error_list = contents['memory_error']

ncoef_list =[]
for coef in coef_list:
    ncoef_list.append(normalize_coefs(coef))

ccoef_list=[]    
for coef in ncoef_list:
    ccoef_list.append(compartment_coef(coef))

#weights go from unscaled, standardised, minmax, normalize
    
unscaled=[]
for coef in ccoef_list:
    unscaled.append(coef[0])
    
unscaled = np.array(unscaled).T

unscaled = unscaled[:,:50]

#delay = np.linspace(0,10000,len(unscaled[0])+1)
delay =np.linspace(0,1000,51)
fig, ax = plt.subplots()
im = ax.imshow(unscaled, cmap="Reds")
#ax.set_xticks(np.arange(len(delay[:-1])))
ax.set_xticklabels([0,0, 200,400,600,800])
ax.set_yticks(np.arange(4))
plt.xlabel("delay (ms)")
plt.ylabel("compartment")
ticks=np.linspace(0.1,0.7,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1%", pad=0.1)
plt.colorbar(im, cax=cax, ticks=ticks)
"""
cbar_kw={}
cbar = ax.figure.colorbar(im, ax=ax, ticks=ticks, **cbar_kw)
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
"""
plt.savefig("us_mem_error_heatmap.png")

sscaled=[]
for coef in ccoef_list:
    sscaled.append(coef[1])
    
sscaled = np.array(sscaled).T

sscaled = sscaled[:,:50]

#delay = np.linspace(0,10000,len(sscaled[0])+1)
delay =np.linspace(0,1000,51)

fig, ax = plt.subplots()
im = ax.imshow(sscaled, cmap="Reds")
#ax.set_xticks(np.arange(len(delay[:-1])))
#ax.set_xticklabels(delay[:-1])
ax.set_xticklabels([0,0, 200,400,600,800])
ax.set_yticks(np.arange(4))
plt.xlabel("delay (ms)")
plt.ylabel("compartment")

#ax.grid(which="minor", color="b", linestyle='-', linewidth=5)


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1%", pad=0.1)

plt.colorbar(im, cax=cax, ticks=ticks)

#cbar_kw={}
#cbar = ax.figure.colorbar(im, ax=ax, ticks=ticks, **cbar_kw)

#plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.savefig("ss_mem_error_heatmap.png")