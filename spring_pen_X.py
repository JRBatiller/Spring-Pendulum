# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:04:20 2018

@author: Joby
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 19:21:02 2018

@author: Joby
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:55:31 2018

@author: Joby
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:42:43 2018

@author: Joby
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 09 15:29:16 2018

@author: Joby
"""

import numpy as np
from scipy import integrate
import math
from scipy.integrate import odeint, ode
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import interpolate
import scipy.io as sio

global xtemp
global ytemp

g=9.81   #super constant

def tri_sin(t, T=1):
    f1 = 2.11
    f2 = 3.73
    f3 = 4.33
    x = np.sin(2*np.pi*f1*t/T)*np.sin(2*np.pi*f2*t/T)*np.sin(2*np.pi*f3*t/T)
    return x

def rotate(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return float(m.T[0]), float(m.T[1])

def resultant_Force(F_list):
    forces = F_list[:,0]
    theta = F_list[:,1]
    forcexy = np.zeros(2)
    
    for i in range(len(F_list)):
        xy = np.array([forces[i]*np.cos(theta[i]),forces[i]*np.sin(theta[i])])
        forcexy=np.vstack((forcexy,xy))
    
    forcexy=forcexy[1:,:] #remove 0,0
    forcexy=np.sum(forcexy,0) # i do not need to remove 0,0
    
    return forcexy

def get_dist_vector(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    dist = np.linalg.norm(p2-p1)
    theta = np.arctan2(y2-y1, x2-x1)
    return dist, theta

def rectangle_pendulum(N_links, base, height=-1.0):

    node_list=[]
    l_node = np.array([-0.5*base,0,0,0]) # x, y, dx, dy
    r_node = l_node + np.array([base,0,0,0])
    node_list.append(l_node)
    node_list.append(r_node)
    #y=np.linspace(0,height,N_links+1)
    y=np.linspace(0,height,N_links)
    for i in range(1,N_links):
        l_node = np.array([-0.5*base,y[i],0,0])
        r_node = np.array([0.5*base,y[i],0,0])
        node_list.append(l_node)
        node_list.append(r_node)
    return node_list
    
def fixed_slope_pendulum(N_links, base, height=-2.0):
    slope =abs( height/(0.5*base))
    node_list=[]
    l_node = np.array([-0.5*base,0,0,0]) # x, y, dx, dy
    r_node = l_node + np.array([base,0,0,0])
    node_list.append(l_node)
    node_list.append(r_node)
    y=np.linspace(0,height,N_links+1)
    x=abs((y-height)/slope)
    for i in range(1,N_links):
        l_node = np.array([-x[i],y[i],0,0])
        r_node = np.array([x[i],y[i],0,0])
        node_list.append(l_node)
        node_list.append(r_node)
    return node_list

def outward_slope_pendulum(N_links, base, height=-2.0):
    slope =-1*abs( height/(0.5*base))
    node_list=[]
    l_node = np.array([-0.5*base,0,0,0]) # x, y, dx, dy
    r_node = l_node + np.array([base,0,0,0])
    node_list.append(l_node)
    node_list.append(r_node)
    y=np.linspace(0,height,N_links+1)
    x=((y+height)/slope)
    for i in range(1,N_links):
        l_node = np.array([-x[i],y[i],0,0])
        r_node = np.array([x[i],y[i],0,0])
        node_list.append(l_node)
        node_list.append(r_node)
    return node_list

def update_nodes(node,m, connections, k, L0, nodes, damping, dt):
    """
    new_node = np.zeros(len(node))
    new_node[0] = node[0]+node[2]*dt 
    new_node[1] = node[1]+node[3]*dt
    new_node[2] = node[2]+ax*dt
    new_node[3] = node[3]+ay*dt
    """
    t = [0, dt]
    new_node = odeint(dnodes, node, t, args=(m, connections, k, L0, nodes, damping))
    
    return new_node[-1]

def dnodes(node, t, m, connections, k, L0, nodes, damping):
    x,y,vx,vy = node
    dx = vx
    dy = vy
    
    force_list=np.zeros(2) #dummy force with 0 amp and dir
    for i in range(len(connections)):
        linked_node=connections[i] #which node we are connected to
        linked_node_coords = nodes[linked_node][0:2]
        L, direction = get_dist_vector(node[0:2], linked_node_coords)
        force=np.array([k[i]*(L-L0[i]),direction])
        force_list=np.vstack((force_list,force))
    
    force_g = np.array([m*g, -np.pi/2])
    damping_x = np.array([-damping*vx, 0 ])
    damping_y = np.array([-damping*vy, np.pi/2 ])
    
    force_list=np.vstack((force_list,force_g, damping_x, damping_y))
    ax, ay = resultant_Force(force_list)/m
    dvx = ax
    dvy = ay
    dnodes = [dx, dy, dvx, dvy]
    return dnodes

def update_springs(nodes, N_links):
    #return 1d array of all springs 
    spring_list = np.zeros(1) 
    spring_angle_list = np.zeros(1)
    for i in range (1, N_links):
        node_0 = nodes[2*i-2][0:2]
        node_1 = nodes[2*i][0:2]
        node_2 = nodes[2*i+1][ 0:2]
        node_3 = nodes[2*i-1][ 0:2]
        l_spring, l_spring_a = get_dist_vector(node_0, node_1)
        m_spring, m_spring_a = get_dist_vector(node_1, node_2)
        r_spring, r_spring_a = get_dist_vector(node_2, node_3)
        ld_spring, ld_spring_a = get_dist_vector(node_0, node_2)
        rd_spring, rd_spring_a = get_dist_vector(node_1, node_3)
        springs = np.hstack((l_spring, m_spring, r_spring, ld_spring, rd_spring))
        angles = np.hstack ((l_spring_a, m_spring_a,r_spring_a, ld_spring_a, rd_spring_a))
        spring_list = np.hstack((spring_list,springs))
        spring_angle_list = np.hstack((spring_angle_list,angles))
    
    spring_list = spring_list[1:]
    spring_angle_list = spring_angle_list[1:]
    return spring_list, spring_angle_list
    
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


def spring_pendulum(N_links, m, k, damping, driving_force, shape, base_length=0.2, ror=0.2):
    
    ###################################################################
    #LISTS
    #node_list = rectangle_pendulum(N_links, base_length, -1)
    #node_list = fixed_slope_pendulum(N_links, base_length,-1)
    #node_list = outward_slope_pendulum(N_links, base_length,-1)
    node_list = shape(N_links, base_length,-1)
    spring_list, spring_angle_list = update_springs(node_list, N_links)
    L0=spring_list # the original spring lengths
    
    dummy_list = np.zeros(len(spring_list))
    master_node_list = node_list
    master_spring_list = np.vstack((dummy_list,spring_list))
    master_spring_angle_list = np.vstack((dummy_list,spring_angle_list))
    master_spring_list=master_spring_list[1:,:] #truncate out dummy list
    master_spring_angle_list=master_spring_angle_list[1:,:] #truncate out dummy list
    new_node_list = node_list
    mass_list = np.ones((2*N_links)) * m
    k_list = np.ones(5*(N_links-1)) * k
    
    """
    #varying k decrease
    k_list=np.array([])
    for i in range(N_links-1):
        spring_set = np.ones(5) * k* (N_links-i)/N_links
        k_list=np.hstack((k_list,spring_set))
    
    #varying k increase
    k_list=np.array([])
    for i in range(1,N_links):
        spring_set = np.ones(5) * k* i/N_links
        k_list=np.hstack((k_list,spring_set))
    """
    
    ####################################################################
    #radius of rotation
    #ror = 0.2
    for i in range(len(node_list)):
        node_list[i] = node_list[i] + np.array([0, -ror, 0, 0])
    
    ######################################################################
    #node coordinates
    node_coords = np.zeros(1) #dummy list
    for nodes in node_list:
        node_coords = np.hstack((node_coords,nodes[0:2]))
    node_coords = node_coords[1:]
    
    #######################################################################
    #PUT DRIVEN NODES HERE
    driven_l_node = node_list[0][0:2]
    driven_r_node = node_list[1][0:2]
    
    #####################################################################
    #connection matrix
    connection_matrix = []
    level=np.array([2,3])
    connection_matrix.append(level)
    connection_matrix.append(level)
    for i in range(1,N_links):
        lower = 2*(i-1)
        upper = min([2*(i+2),len(node_list)])
        level=np.arange(lower,upper)
        connection_matrix.append(np.delete(level,2))
        connection_matrix.append(np.delete(level,3))
     
    #########################################################################
    #the main simulation
    
    iteration = 0
    for theta in driving_force:
    
        if iteration%5000==0:
            print("iteration is {} of {}".format(iteration,len(driving_force)))
        iteration+=1
        
        coords=np.zeros(1)#dummy list
        
        for i in range(N_links):
            top_l = 5*(i-1)
            top_ld = 5*(i-1) + 4 
            mid = 5*(i-1) +1
            bot_l = 5*(i)
            bot_ld = 5*(i) + 3 # yes this is slightly off but true
            
            top_r = 5*(i-1) + 2
            top_rd = 5*(i-1) + 3
            #mid
            bot_r = 5*(i) + 2
            bot_rd = 5*(i) + 4 # yes this is slightly off but true
            if i==0 : 
                #left node   
                x,y = rotate(driven_l_node,theta)
                new_node_list[0] = np.array([x,y,0,0]) 
                
                #right node    
                x,y = rotate(driven_r_node,theta)
                new_node_list[1] = np.array([x,y,0,0])
                
            else:
                
                if i==N_links-1:
                    l_index = [top_l, top_ld, mid] 
                    r_index = [top_rd, top_r, mid]                    
                else:
                    l_index = [top_l, top_ld, mid, bot_l, bot_ld] 
                    r_index = [top_rd, top_r, mid, bot_rd, bot_r]
                
                L_k = k_list[l_index]
                R_k = k_list[r_index]
                L_L0 = L0[l_index]
                R_L0 = L0[r_index]
                #left node
                new_node_list[2*i] = update_nodes(node_list[2*i], mass_list[2*i], connection_matrix[2*i], L_k, L_L0, node_list, damping, dt)
                #right node
                new_node_list[2*i+1] = update_nodes(node_list[2*i+1], mass_list[2*i+1], connection_matrix[2*i+1], R_k, R_L0, node_list, damping, dt)
                
            
        new_spring_list, new_spring_angle_list = update_springs(new_node_list, N_links)
        master_spring_list=np.vstack((master_spring_list,new_spring_list))
        master_spring_angle_list=np.vstack((master_spring_angle_list,new_spring_angle_list))
        
        for nodes in new_node_list:
            coords = np.hstack((coords,nodes[0:2]))
        coords = coords[1:]
        node_coords = np.vstack((node_coords,coords))
        
        node_list = new_node_list
        master_node_list=np.vstack((master_node_list,node_list))
        
    return master_spring_list, master_spring_angle_list, node_coords, master_node_list
    




##################################################################
#parameters
     
base_length=0.2
#base_length=0.1

#damping = 0.5
#k = 10
#k_list = [0.1, 1, 10, 25, 50, 100, 150, 200, 1000]
#d_list = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 10]
#k_list=[1, 10, 100, 1000]
#d_list=[0.25, 0.5, 1, 2]
#k_list = [1, 5, 10, 25, 50, 100, 150, 200, 1000]
#d_list = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 5, 10]
k_list=[1]
d_list=[1]

m=0.1

N_links = 5

total_iter=len(k_list)*len(d_list)

########################################################################
#make input function
t0=0
t_end=100
dt = 0.001
    
time = np.linspace(t0, t_end, int(t_end/dt))
T=10
driving_force = 0.7*tri_sin(time,T)
# 5 seconds to make pendulum settle
settle_time = np.zeros(int(5/dt))
driving_force = np.hstack((settle_time,driving_force))
shape=fixed_slope_pendulum
########################################################################   
iteration=0
for k in k_list:
    for damping in d_list:
        iteration = iteration + 1                   
        print "running parameter set ({},{}) {} of {}".format(k,damping,iteration,total_iter)
        master_spring_list, master_spring_angle_list, node_coords, node_list = spring_pendulum(N_links, m, k, damping, driving_force, shape, base_length=base_length)
        name = 'N_'+ str(N_links) + '_m_'+str(m)+'_k_'+str(k)+'_d_'+str(damping)+'.mat'
        sio.savemat("spring_pendulum_X_"+name, {'lengths':master_spring_list, 'angles':master_spring_angle_list, 'nodes':node_coords, 'input':driving_force})            

##########################################################################
#anim = animate_pendulum(node_coords, driving_force, 10, 0)
#anim.save('spring_pendulum_X.mp4', fps=100, extra_args=['-vcodec', 'libx264'])    