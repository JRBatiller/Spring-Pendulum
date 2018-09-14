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
#THIS CODE FOR MEMORY TESTS

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

def tri_sin(t, T=1):
    f1 = 2.11
    f2 = 3.73
    f3 = 4.33
    x = np.sin(2*pi*f1*t/T)*np.sin(2*pi*f2*t/T)*np.sin(2*pi*f3*t/T)
    return x

def narma_series(u, n):
    
    y=np.zeros(len(u))
    for i in range(n+1,len(u)):			
        y[i] = 0.3*y[i-1] + 0.05*y[i-1]*sum(y[i-n:i-1])+1.5*u[i-n]*u[i-1]+0.1
        #y(k,1) = 0.3*y(k-1,1) + 0.05*y(k-1,1)*(sum(y(k-n:k-1,1))) + 1.5*u(k-n,1)*u(k-1,1) + 0.1;
    
    return y

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


def get_xy_coords(p, N_links=5):
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

def animate_pendulum_X(p, t, weights=[], skip=1, plot=0):
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
    
    if not np.size(weights):
        colors = ['blue' for i in range(N_links*5)]
        colors = ['black','black','black'] + colors
    
    else:
        weights = abs(weights)
        weights = weights / max(weights)
        colors = [cm.jet(i) for i in weights]
        colors = ['black','black','black'] + colors
    
    lines = []
    for index in range(total):
        """
        if index <3:
            colors = "black"
        else:
            colors = "blue"
        """
        lobj = ax.plot([], [], 'o-', lw=2, color=colors[index])[0]
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
    
    if plot==0:
        plt.close(fig)
    return anim


def animate_pendulum_X_C(p, t, colors=[], skip=1, plot=0):
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
    
    if not np.size(colors):
        colors = ['blue' for i in range(N_links*5)]
        colors = ['black','black','black'] + colors
    
    else:
        colors=list(colors)
        colors = ['black','black','black'] + colors
    
    lines = []
    for index in range(total):
        """
        if index <3:
            colors = "black"
        else:
            colors = "blue"
        """
        lobj = ax.plot([], [], 'o-', lw=2, color=colors[index])[0]
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
    
    if plot==0:
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
    
    nscaler = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    nX_train=nscaler.transform(X_train)
    nX_test=nscaler.transform(X_test)
    
    norm_training = preprocessing.normalize(training, axis=0, norm='max')
    norm_X_train=norm_training[washout+delay:train_len+delay]
    norm_X_test=norm_training[train_len+delay:train_len+test_len+delay]  
    
    lm=linear_model.LinearRegression()
    lm.fit(X_train,Y_train)
    pred_train=lm.predict(X_train)
    pred_test=lm.predict(X_test)    
    
    if plot==1:  
        colors = np.vstack((cm.Blues(np.linspace(0.5, 1, 5)),cm.Greens(np.linspace(0.5, 1, 5)),cm.Oranges(np.linspace(0.5, 1, 5)),cm.Reds(np.linspace(0.5, 1, 5))))
        plt.figure()
        
        for row,c in zip(X_train[2000:12000].T, colors):
            plt.plot(range(2000,12000),row, color=c, linewidth=2.5)
        
        plt.xlabel("time (ms)")
        plt.ylabel("spring lengths (m)")
        
        #plt.plot(Y_train[2000:12000])
        if save==1:
            plt.savefig("spring_lengths.png")
        
    
        plt.figure()
        plt.plot(range(2000,12000),pred_test[:10000], linewidth=3.0)
        plt.plot(range(2000,12000),Y_test[:10000], 'r--', linewidth=2.5)
        plt.xlabel("time (ms)")
        plt.ylabel("y(t)/O(t)")
        
        plt.legend(['system output', 'target output'])
        if save==1:
            plt.savefig("spring_output.png")
    
    slm=linear_model.LinearRegression()
    slm.fit(sX_train,Y_train)
    spred_train=slm.predict(sX_train)
    spred_test=slm.predict(sX_test)    
    
    nlm=linear_model.LinearRegression()
    nlm.fit(nX_train,Y_train)
    npred_train=nlm.predict(nX_train)
    npred_test=nlm.predict(nX_test)    

    norm_lm=linear_model.LinearRegression()
    norm_lm.fit(norm_X_train,Y_train)
    norm_pred_train=norm_lm.predict(norm_X_train)
    norm_pred_test=norm_lm.predict(norm_X_test)    


    error = mean_squared_error(Y_test, pred_test)    
    serror= mean_squared_error(Y_test, spred_test)
    nerror= mean_squared_error(Y_test, npred_test)
    norm_error = mean_squared_error(Y_test, norm_pred_test)
    #print(mean_squared_error(pred_test,Y_test))
    
    coefs = [lm.coef_, slm.coef_, nlm.coef_, norm_lm.coef_]
    errors = [error, serror,nerror, norm_error]
    return coefs, errors

def normalize_coefs(coefs):
    #should be a list
    normed_coefs=[]
    for coef in coefs:
        coef = abs(coef)
        coef = coef/max(coef)
        normed_coefs.append(coef)
    return normed_coefs

plt.rcParams.update({'font.size': 22})
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

contents=sio.loadmat('NARMA10_2.mat')
narma10_2=contents['dat']['y'][0][0]
narma10_2n=contents['dat']['yn'][0][0]
narma10_2_input=contents['dat']['u'][0][0]

dt=0.001
end=300.0
N=int(end/dt)
washout=int(5/dt)+5000

train_len=washout+50000
test_len=15000

t = np.linspace(0.0, end, N)

#spring_pen_set, file_names = get_output("Spring Pendulum X/A=0.7 T=10 ror=0.2")
spring_pen_set, file_names = get_output("Spring Pendulum X/all")
#spring_pen_set, file_names = get_output("Spring Pendulum X/rectangle")
#spring_pen_set, file_names = get_output("Spring Pendulum X/T=5")

spring_pen_set, input_signal, nodes_list = get_spring_lengths(spring_pen_set)

error_list=np.zeros(4)
error_list_all=np.zeros(4)

#22 is k100 d1
#24 is k100, d1.5
#31 is k10 d 1
#33 is k10, d1.5

total_sets = len(file_names)

#for set_number in range(total_sets):
for set_number in [22]:

    L = spring_pen_set[set_number]
    L_input = input_signal[set_number][0]
    L_name = file_names[set_number] 
    L_nodes = nodes_list[set_number]
    
    print(L_name)
                
    _,_,_,_,N_links,_,mass,_,k,_,d = L_name.split("_")
    d = d[:-4]
    N_links = int(N_links)
    mass = float(mass)
    k = float(k)
    d = float(d)    
            
    n_delay = 6000
    #N2_coefs, N2_errors = LinReg(L, narma2_10, train_len, test_len, n_delay, washout=washout, plot=0)
    #print(N2_errors[0])
    #N5_coef,N5_error = LinReg(L, narma5_10, train_len, test_len, n_delay, washout=washout, plot=1, save=1)
    #print(N5_error[0])
    N10_coef,N10_error = LinReg(L, narma10_10, train_len, test_len, n_delay, washout=washout, plot=1, save=1)
    #N10_coef,N10_error = LinReg(L, narma10_5, train_len, test_len, n_delay, washout=washout, plot=1)
    
    print(N10_error[0])
    #N15_coef,N15_error = LinReg(L, narma15_10, train_len, test_len, n_delay, washout=washout, plot=1)
    
    #N2_coef_scale,N2_error_scale = LinReg(L, narma2_10, train_len, test_len, n_delay, washout=washout, scale=1, plot=1,save=1)
    
    
    v_delay = 4800
    V_coefs, V_errors = LinReg(L, volterra_10, train_len, test_len, v_delay,washout=washout, plot=0)
    print(V_errors[0])
    X_nodes = L_nodes[washout:train_len]
    
    #normed_coef = abs(SN2_coef)
    #normed_coef = normed_coef/max(normed_coef)
    
    #print(normed_coef.reshape((N_links-1,5)))
    anim_len = np.linspace(0,100, len(X_nodes))
    #animate_pendulum_X(X_nodes, anim_len, skip=10, plot=1)
    
    
"""    
    #################################################
    #Memory Test
    
    coef_list=[]
    mem_error=[]
    test_range=np.arange(0,10000,20)
    
    for i in test_range:
        dummy = np.zeros(i)
        
        delayed_input = np.hstack((dummy,L_input))[:len(L_input)]
        delayed_output = narma_series(0.2*delayed_input, 10)
        #delayed_output = np.hstack((dummy,L_input))
        #delayed_output = delayed_output[:len(L_input)]
        #delayed_output = (delayed_output**3 + delayed_output +0.1)*L_input
        
        if i%500==0:
            coefs, errors = LinReg(L, delayed_output, train_len, test_len, 0, washout=washout, plot=0)
            #plt.savefig("mem_error_"+str(i)+".png")
        else:
            coefs, errors = LinReg(L, delayed_output, train_len, test_len, 0, washout=washout)
        
        
        #error = mean_squared_error(L_input, delayed_output)
        mem_error.append(errors)
        coef_list.append(coefs)
        
    sio.savemat("memory_error_"+str(k)+"_"+str(d), {'memory_error':mem_error, 'coef_list':coef_list})            

    #plt.figure()
    #plt.plot(test_range, mem_error)
    #plt.savefig("memory_test_"+str(k)+"_"+str(d)+".png")
"""