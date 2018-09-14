# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 23:46:08 2018

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


pi=np.pi

def get_output(directory):
    dataDir = "C:/Users/Joby/Documents/MSc Summer Project/"+directory+"/"
    mats = []
    file_name_list=os.listdir( dataDir )
    for file in file_name_list :
        mats.append( sio.loadmat( dataDir+file ) )
    
    return mats, file_name_list


def get_ave_error_d(directory):
    mats, file_names = get_output(directory)
    xd=[]
    for mat in mats:
        xd.append(mat["memory_error"][:,0])
    xd=np.array(xd)
    xd=xd[:,:50]
    ave_xd = np.mean(xd,1)

    d_list=[]
    for name in file_names:
        _,_,_,d=name.split("_")
        d_list.append(float(d[:-4]))
    d_list=np.array(d_list)
    return(ave_xd, d_list)

def get_ave_error_k(directory):
    mats, file_names = get_output(directory)
    xk=[]
    for mat in mats:
        xk.append(mat["memory_error"][:,0])
    xk=np.array(xk)
    xk=xk[:,:50]
    ave_xk = np.mean(xk,1)

    k_list=[]
    for name in file_names:
        _,_,k,_=name.split("_")
        k_list.append(float(k))
    k_list=np.array(k_list)
    return(ave_xk, k_list)

    
    
plt.rcParams.update({'font.size': 18})
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

xd100, d_list = get_ave_error_d("Spring Pendulum X/memory test/vary d/100")
xd200, _ = get_ave_error_d("Spring Pendulum X/memory test/vary d/200")
xd10, _ = get_ave_error_d("Spring Pendulum X/memory test/vary d/10")
xd50, _ = get_ave_error_d("Spring Pendulum X/memory test/vary d/50")


mask = np.ones(len(d_list), dtype=bool)
mask[[0,7, 9]]=False

plt.figure()
plt.scatter(d_list[mask],xd100[mask], s=100, c="r", marker="o")
plt.plot(np.unique(d_list[mask]), np.poly1d(np.polyfit(d_list[mask], xd100[mask], 1))(np.unique(d_list[mask])), color="#B22222")
plt.scatter(d_list[mask],xd50[mask], s=100, c="g", marker="^")
plt.plot(np.unique(d_list[mask]), np.poly1d(np.polyfit(d_list[mask], xd50[mask], 1))(np.unique(d_list[mask])), color="#228B22")
plt.scatter(d_list[mask],xd10[mask], s=100, c="b", marker="s")
plt.plot(np.unique(d_list[mask]), np.poly1d(np.polyfit(d_list[mask], xd10[mask], 1))(np.unique(d_list[mask])), color="#1874CD")

#plt.scatter(d_list[mask],xd200[mask], s=100, c="g", marker="^")
#plt.plot(np.unique(d_list[mask]), np.poly1d(np.polyfit(d_list[mask], xd200[mask], 1))(np.unique(d_list[mask])), color="green")
plt.legend(["k=100","k=50","k=10"])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim(0,0.0001)
plt.xlabel("damping coefficient (Ns/m)")
plt.ylabel("mean squared error")
plt.savefig("memory_ave_vary_d.png")


xk1, k_list = get_ave_error_k("Spring Pendulum X/memory test/vary k/1.0")
xk1_5, _ = get_ave_error_k("Spring Pendulum X/memory test/vary k/1.5")
xk2, _ = get_ave_error_k("Spring Pendulum X/memory test/vary k/2.0")
xk_25, _ = get_ave_error_k("Spring Pendulum X/memory test/vary k/0.25")
xk_5, _ = get_ave_error_k("Spring Pendulum X/memory test/vary k/0.5")


mask = np.ones(len(k_list), dtype=bool)
mask[[0, 1,4]]=False

plt.figure()
plt.scatter(k_list[mask],xk_25[mask],s=100, c="r", marker="o")
plt.plot(np.unique(k_list[mask]), np.poly1d(np.polyfit(k_list[mask], xk_25[mask], 2))(np.unique(k_list[mask])), color="#B22222")
#plt.scatter(k_list[mask],xk_5[mask],s=100, c="r", marker="o")
#plt.plot(np.unique(k_list[mask]), np.poly1d(np.polyfit(k_list[mask], xk_5[mask], 2))(np.unique(k_list[mask])), color="#B22222")

plt.scatter(k_list[mask],xk1[mask],s=100, c="g", marker="^")
plt.plot(np.unique(k_list[mask]), np.poly1d(np.polyfit(k_list[mask], xk1[mask], 2))(np.unique(k_list[mask])), color="#228B22")
plt.scatter(k_list[mask],xk2[mask],s=100, c="b", marker="s")
plt.plot(np.unique(k_list[mask]), np.poly1d(np.polyfit(k_list[mask], xk2[mask], 2))(np.unique(k_list[mask])), color="#1874CD")
plt.legend(["d=0.25","d=1.0","d=2.0"])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylim(0,0.0001)
plt.xlabel("spring constant (N/m)")
plt.ylabel("mean squared error")
plt.savefig("memory_ave_vary_k.png")