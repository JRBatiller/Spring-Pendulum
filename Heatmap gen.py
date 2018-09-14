# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 22:09:57 2018

@author: Joby
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 17:01:44 2018

@author: Joby
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 21:11:03 2018

@author: Joby
"""
#THIS CODE GENERATES HEATMAPS

from scipy.integrate import odeint
from scipy.fftpack import fft, fftfreq, fftshift
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.plotly as py
import plotly.graph_objs as go

import sklearn
import sklearn.cross_validation
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


pi=np.pi

def tri_sin(t, T=1):
    f1 = 2.11
    f2 = 3.73
    f3 = 4.33
    x = np.sin(2*pi*f1*t/T)*np.sin(2*pi*f2*t/T)*np.sin(2*pi*f3*t/T)
    return x

def get_rel_angles(abs_angles, N):

    y = abs_angles

    for i in range(N):
        if i == 0:
            angles = y[:, i]
            vels = y[:, i + N]
        else:
            angles = np.vstack((angles, y[:, i] - y[:, i - 1]))
            vels = np.vstack((vels, y[:, i + N] - y[:, i + N - 1]))

    return np.vstack((angles, vels))


def get_L_matrix(mats,N=4):
    
    L_matrix_list=[]
    driving_input_list=[]
    for output in mats:
        
                
        
        L = output['output']
        if N:
            N_links=N
        else:
            x,N_links=np.shape(L)
            N_links=N_links/2
        L = get_rel_angles(L, N_links)
        driving_input_list.append(L[0])
        L=L[1:N_links]
        #L = np.vstack([L,np.ones(len(L[0]))])
        L=L.T    
        L_matrix_list.append(L)
    
    return L_matrix_list, driving_input_list

def get_signal_freq(signal, dt, plot=0):

    yf = fft(signal)
    N=np.size(signal)
    freq=fftfreq(N,dt)
    ind_freq=np.arange(1,N/2+1)
    psd=abs(yf[ind_freq]**2)+abs(yf[-ind_freq]**2)
    #print np.where(psd>0.1*np.max(psd))
    index,=np.where(psd>0.1*np.max(psd))
    if plot:
        plt.figure()
        plt.plot(freq[ind_freq],psd/N,'k-')
        plt.ion()
        plt.show()
        
    
        
    return freq[ind_freq[index]]

def get_output(directory):
    dataDir = "C:/Users/Joby/Documents/MSc Summer Project/"+directory+"/"
    mats = []
    file_name_list=os.listdir( dataDir )
    for file in file_name_list :
        mats.append( sio.loadmat( dataDir+file ) )
    
    return mats, file_name_list

def get_spring_lengths(mats):
    spring_length_list=[]
    input_list=[]
    nodes_list=[]
    for output in mats:
        spring_length_list.append(output['lengths'])
        input_list.append(output['input'])
        nodes_list.append(output['nodes'])
    return spring_length_list, input_list, nodes_list


def get_xy_coords(p, N_links=4):
    """Get (x, y) coordinates from generalized coordinates p"""
    x = p[:,::2]
    y = p[:,1::2]
    return x, y


def animate_pendulum(p, t, skip=1, see_plot=0):
    #t = np.linspace(0, 10, 200)
    #p = integrate_pendulum(n, t)
    N_links = int(len(p[0])/4)
    x, y = get_xy_coords(p, N_links)
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    x=x[::skip]
    y=y[::skip]
    
    origin = np.zeros((len(x),1))
    
    xlist=[]
    ylist=[]
    
    #draw lines from origin to first 2 nodes
    xs=np.hstack([origin,x[:,0:1]])
    xlist.append(xs)
    xs=np.hstack([origin,x[:,1:2]])
    xlist.append(xs)
    xs=np.hstack([x[:,0:1],x[:,1:2]])
    xlist.append(xs)
    ys=np.hstack([origin,y[:,0:1]])
    ylist.append(ys)
    ys=np.hstack([origin,y[:,1:2]])
    ylist.append(ys)
    ys=np.hstack([y[:,0:1],y[:,1:2]])
    ylist.append(ys)
    
    #old pendulum without X
    #draw horizontal springs
    for N in range(N_links):
        
        xs = x[:,2*N:2*N+2]
        ys = y[:,2*N:2*N+2]
        xlist.append(xs)
        ylist.append(ys)            
    
    #print(1000/skip * t.max() / len(t))
    xl = x[:,::2]
    xr = x[:,1::2]
    yl = y[:,::2]
    yr = y[:,1::2]
    #draw vertical springs
    for N in range(1,N_links):
        xs = xl[:,N-1:N+2]
        xlist.append(xs)
        ys = yl[:,N-1:N+2]
        ylist.append(ys)            
        xs = xr[:,N-1:N+2]
        xlist.append(xs)
        ys = yr[:,N-1:N+2]
        ylist.append(ys)            
    
    
    total = 2*(N_links-1) + N_links +2
    
    line, = ax.plot([], [], 'o-', lw=2)
    
    lines = []
    for index in range(total):
        if index <3:
            colors = "black"
        else:
            colors = "blue"
        lobj = ax.plot([], [], 'o-', lw=2, color=colors)[0]
        lines.append(lobj)
    
    def init():
        #line.set_data([], [])
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(i):
        """
        line.set_data(x[i], y[i]+2)
        return line,
        """
        
        #for index in range(0,1):
        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum][i], ylist[lnum][i]+1) # set data for each line separately. 

        return lines
        
    anim = FuncAnimation(fig, animate, frames=len(t)/skip,
                                   interval=1,
                                   blit=True, init_func=init)
    #1000/skip * t.max() / len(t)
    
    if see_plot==0:
        plt.close(fig)
    return anim

def animate_pendulum_X(p, t, skip=1, see_plot=0):
    #t = np.linspace(0, 10, 200)
    #p = integrate_pendulum(n, t)
    N_links = int(len(p[0])/4)
    x, y = get_xy_coords(p, N_links)
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    x=x[::skip]
    y=y[::skip]
    
    origin = np.zeros((len(x),1))
    
    xlist=[]
    ylist=[]
    
    #draw lines from origin to first 2 nodes
    xs=np.hstack([origin,x[:,0:1]])
    xlist.append(xs)
    xs=np.hstack([origin,x[:,1:2]])
    xlist.append(xs)
    xs=np.hstack([x[:,0:1],x[:,1:2]])
    xlist.append(xs)
    
    ys=np.hstack([origin,y[:,0:1]])
    ylist.append(ys)
    ys=np.hstack([origin,y[:,1:2]])
    ylist.append(ys)
    ys=np.hstack([y[:,0:1],y[:,1:2]])
    ylist.append(ys)
    
    
    for i in range(N_links-1):
        xs=np.hstack([x[:,2*i+0:2*i+1],x[:,2*i+2:2*i+3]])
        xlist.append(xs)
        xs=np.hstack([x[:,2*i+2:2*i+3],x[:,2*i+3:2*i+4]])
        xlist.append(xs)
        xs=np.hstack([x[:,2*i+3:2*i+4],x[:,2*i+1:2*i+2]])
        xlist.append(xs)
        xs=np.hstack([x[:,2*i+0:2*i+1],x[:,2*i+3:2*i+4]])
        xlist.append(xs)
        xs=np.hstack([x[:,2*i+2:2*i+3],x[:,2*i+1:2*i+2]])
        xlist.append(xs)
        ys=np.hstack([y[:,2*i+0:2*i+1],y[:,2*i+2:2*i+3]])
        ylist.append(ys)
        ys=np.hstack([y[:,2*i+2:2*i+3],y[:,2*i+3:2*i+4]])
        ylist.append(ys)
        ys=np.hstack([y[:,2*i+3:2*i+4],y[:,2*i+1:2*i+2]])
        ylist.append(ys)        
        ys=np.hstack([y[:,2*i+0:2*i+1],y[:,2*i+3:2*i+4]])
        ylist.append(ys)
        ys=np.hstack([y[:,2*i+2:2*i+3],y[:,2*i+1:2*i+2]])
        ylist.append(ys)
        
    total = 5*(N_links-1)+3
    
    global xtemp
    xtemp = xlist
    global ytemp
    ytemp =ylist
    
    line, = ax.plot([], [], 'o-', lw=2)
    
    lines = []
    for index in range(total):
        if index <3:
            colors = "black"
        else:
            colors = "blue"
        lobj = ax.plot([], [], 'o-', lw=2, color=colors)[0]
        lines.append(lobj)
    
    def init():
        #line.set_data([], [])
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(i):
        """
        line.set_data(x[i], y[i]+2)
        return line,
        """
        
        #for index in range(0,1):
        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum][i], ylist[lnum][i]+1) # set data for each line separately. 

        return lines
        
    anim = FuncAnimation(fig, animate, frames=len(t)/skip,
                                   interval=1,
                                   blit=True, init_func=init)
    #1000/skip * t.max() / len(t)
    
    if see_plot==0:
        plt.close(fig)
    return anim


def LinReg(training, teacher, train_len, test_len, delay, washout=8000, plot=0, save=0):    
    
    #Y_input = narma_10_input[washout:train_len].ravel()
    Y_train = teacher[washout:train_len].ravel()
    Y_test = teacher[train_len:train_len+test_len].ravel()

    #X_input=input_signal[set_number][0][washout+delay:train_len+delay]
    X_train=training[washout+delay:train_len+delay]
    X_test=training[train_len+delay:train_len+test_len+delay]  
    #X_nodes = nodes_list[set_number][washout+delay:train_len+delay]
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    sX_train=scaler.transform(X_train)
    sX_test=scaler.transform(X_test)
    
    lm=linear_model.LinearRegression()
    lm.fit(X_train,Y_train)
    pred_train=lm.predict(X_train)
    pred_test=lm.predict(X_test)    
    
    if plot==1:  
        """
        plt.figure()
        plt.plot(X_train[2000:12000])
        plt.plot(Y_train[2000:12000])
        if save==1:
            plt.savefig("lengths.png")
        """
    
        plt.figure()
        plt.plot(pred_test)
        plt.plot(Y_test)
        if save==1:
            plt.savefig("spring_output.png")
    
    slm=linear_model.LinearRegression()
    slm.fit(sX_train,Y_train)
    spred_train=slm.predict(sX_train)
    spred_test=slm.predict(sX_test)    
    


    error = mean_squared_error(Y_test, pred_test)    
    #print(mean_squared_error(pred_test,Y_test))
    return lm.coef_, slm.coef_, error
        

def make_3d_plot(error_list):    
    xdata = error_list[:,0]
    ydata = error_list[:,1]
    zdata = error_list[:,2]
    vdata = error_list[:,3]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens');    
    
    ax.set_xlabel("spring k")
    ax.set_ylabel("damping")
    #ax.savefig("narma_error.png")
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter3D(xdata, ydata, vdata,  cmap='Greens');    
    ax2.set_xlabel("spring k")
    ax2.set_ylabel("damping")
    #ax2.savefig("colterra_error.png")

def make_heatmap_plot(error_list, save=0):
    ###################
    #HEATMAP SECTION
    xdata = error_list[:,0]
    from collections import OrderedDict
    xdata = error_list[:,0]
    ydata = error_list[:,1]
    zdata = error_list[:,2]
    vdata = error_list[:,3]
    
    
    x_data = np.array(list(OrderedDict.fromkeys(xdata)))  
    y_data = np.array(list(OrderedDict.fromkeys(ydata)))
    x_index = np.argsort(x_data) # get indexes of sorted array
    y_index = np.argsort(y_data)    
    x_names = sorted(y_data) #YES THIS IS INVERTED X IS COLUMNS!!!
    y_names = sorted(x_data)
    x_names = [str(i) for i in x_names]
    y_names = [str(i) for i in y_names]
        
    z_val=np.reshape(zdata, (len(x_names),len(y_names)))
    ######## THIS AREA IS FARLY SPECIFIC NEED TO CHANGE THIS PER FOLDER
    # IT HINK I FIXED IT
    z_val=np.vstack((z_val[x_index])) #rearrange to ascending order
    z_val=np.vstack((z_val.T[y_index])).T #rearrange to ascending order
    
    
    # log just to make differences smaller
    z_val = np.log10(z_val)
    
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    #for narma2 maxmin = (-1,-4.5)
    #for narma 5 maxmin = (-4,-7)
    #for narma 10 maxmin = (-4,-6.5)
    #for volterra maxmin = (0.5, -2.5)
    im = ax.imshow(z_val, vmin=-6.5, vmax=-4)
    #im = ax.imshow(z_val, vmin=-6, vmax=-3)
    
    cbar_kw={}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel("mean squared error (log)", rotation=-90, va="bottom")
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_names)))
    ax.set_yticks(np.arange(len(y_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_names)
    ax.set_yticklabels(y_names)
    ax.set_xlabel("damping coefficient (Ns/m)")
    ax.set_ylabel("spring constant (N/m)")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
    # Loop over data dimensions and create text annotations.
        
    for i in range(len(y_names)):
        for j in range(len(x_names)):
            text = ax.text(j, i, round(z_val[i, j],2),
                           ha="center", va="center", color="w")
    
    ax.set_title("Error Heatmap")
    fig.tight_layout()
    plt.show()
    if save:
        name = "error_heatmap"
        if isinstance(save, basestring):
            name =  name+'_'+save
            
        plt.savefig(name+".png")


    
#############################################################################
#start
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

contents=sio.loadmat('volterra.mat')
volterra=contents['dat']['y'][0][0]
volterra_input=contents['dat']['u'][0][0]
volterra_input=volterra_input/max(volterra_input)

contents=sio.loadmat('volterra_10.mat')
volterra_10=contents['dat']['y'][0][0]
volterra_10_input=contents['dat']['u'][0][0]
volterra_10_input=volterra_10_input/max(volterra_10_input)

contents=sio.loadmat('NARMA2_10.mat')
narma2_10=contents['dat']['y'][0][0]
narma2_10n=contents['dat']['yn'][0][0]
narma2_10_input=contents['dat']['u'][0][0]

contents=sio.loadmat('NARMA5_10.mat')
narma5_10=contents['dat']['y'][0][0]
narma5_10n=contents['dat']['yn'][0][0]
narma5_10_input=contents['dat']['u'][0][0]

contents=sio.loadmat('NARMA10_10.mat')
narma10_10=contents['dat']['y'][0][0]
narma10_10n=contents['dat']['yn'][0][0]
narma10_10_input=contents['dat']['u'][0][0]

contents=sio.loadmat('NARMA15_10.mat')
narma15_10=contents['dat']['y'][0][0]
narma15_10n=contents['dat']['yn'][0][0]
narma15_10_input=contents['dat']['u'][0][0]

contents=sio.loadmat('NARMA10_5.mat')
narma10_5=contents['dat']['y'][0][0]
narma10_5n=contents['dat']['yn'][0][0]
narma10_5_input=contents['dat']['u'][0][0]

dt=0.001
end=300.0
N=int(end/dt)
washout=int(5/dt)+5000

train_len=washout+50000
test_len=15000

t = np.linspace(0.0, end, N)

#spring_pen_set, file_names = get_output("Spring Pendulum X/all")

#dir_list = ["all","all - Copy", "bl=0.1","m=0.01","m=0.5","rectangle","varying k 2","increasing k"]
dir_list = ["outward small"]
#dir_list = ["T=5"]

for directory in dir_list:
    spring_pen_set, file_names = get_output("Spring Pendulum X/"+directory)
    
    spring_pen_set, input_signal, nodes_list = get_spring_lengths(spring_pen_set)
    
    error_list=np.zeros(4)
    
    for set_number in range(len(spring_pen_set)):
        L = spring_pen_set[set_number]
        L_name = file_names[set_number] 
        print(L_name)
                
        _,_,_,_,N_links,_,mass,_,k,_,d = L_name.split("_")
        d = d[:-4]
        N_links = int(N_links)
        mass = float(mass)
        k = float(k)
        d = float(d)    
            
        n_delay = 6000
        #N_coef,SN_coef, N_error = LinReg(L, narma10_10, train_len, test_len, n_delay)
        N_coef,SN_coef, N_error = LinReg(L, narma10_10, train_len, test_len, n_delay)
        
        v_delay = 4800
        V_coef,SV_coef, V_error = LinReg(L, volterra_10, train_len, test_len, v_delay)
        
        errors = np.array([k,d,N_error,V_error])
        
        
        error_list = np.vstack((error_list,errors))
        
    error_list = error_list[1:,:]
    
    #make_3d_plot(error_list)
    make_heatmap_plot(error_list, save=directory)
    
    #animate_pendulum(X_nodes, X_train, 10, 1)
