"""
This file contains all the classes and functions that make up the ktg_python codebase.
"""

import os
import sys
import time
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#### PARALLELIZE FUNCTIONS ####
def _adjust_velocity(mol_pair):
    m_1=mol_pair[0]
    m_2=mol_pair[1]
    
    x_1=m_1.get_position()
    x_2=m_2.get_position()
    
    u_1=m_1.get_velocity()
    u_2=m_2.get_velocity()
    
    ms_1=m_1.get_mass()
    ms_2=m_2.get_mass()
    
   
    v_1=u_1 - 2*ms_2/(ms_1 + ms_2) * (np.dot(u_1-u_2,x_1-x_2)/np.linalg.norm(x_1-x_2)**2) * (x_1 - x_2)
    v_2=u_2 - 2*ms_1/(ms_1 + ms_2) * (np.dot(u_2-u_1,x_2-x_1)/np.linalg.norm(x_2-x_1)**2) * (x_2 - x_1)
    
    
    m_1.set_velocity(v_1)
    m_2.set_velocity(v_2)


#### CLASSES ####
class Molecule(object):
    #Constructor
    def __init__(self,molecule,position,velocity,mass,color="black"):
        self.molecule=molecule
        self.position=np.array([x_i for x_i in position])
        self.velocity=np.array([v_i for v_i in velocity])
        self.mass=mass
        self.color=color
       
    #Setters for position, velocity, mass and color
    def set_position(self,position):
        self.position=np.array([x_i for x_i in position])
        
    def set_velocity(self,velocity):
        self.velocity=np.array([v_i for v_i in velocity])
        
    def set_color(self,color):
        self.color=color
    
    def set_mass(self,mass):
        self.mass=mass
        
    #Getters for position, velocity, mass and color
    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
        
    def get_color(self):
        return self.color
        
    def get_mass(self):
        return self.mass
    
class Simulation(object):
    #Constructor
    def __init__(self,name,box_dim,t_step,particle_radius):
        #Set simulation inputs
        self.name=name #Name of the simulation
        self.box_dim=[x for x in box_dim] #Dimensions of the box
        self.t_step=t_step #Timestep
        self.particle_radius=particle_radius #Radius of the particles
        
        #Calculate volume and number of dimensions
        self.V=np.prod(self.box_dim) #Area/Volume of the box
        self.dim=int(len(box_dim)) #Number of dimensions (2D or 3D)
        
        #Initialize paramters
        self.molecules=[] #Create empty list to store objects of class Molecule
        self.n_molecules=0 #Create variable to store number of molecules
        self.wall_collisions=0 #Create variable to store number of wall collisions
        self.wall_momentum=0 #Create variable to store net momentum exchanged with wall
        
    def _generate_initial_positions(self,n,dist="uniform"):
        #Uniform distribution
        if dist=="uniform":
            _pos=np.random.uniform(low=[0]*self.dim,high=self.box_dim,size=(n,self.dim))
            
        #Store positions in temporary variable    
        self._positions=_pos

    def _generate_initial_velocities(self,n,v_mean,v_std,dist="normal"):
        #Normal distribution with mean v_mean and std v_std
        if dist=="normal":
            self.v_mean=v_mean
            self.v_std=v_std
            _vel=np.random.normal(loc=v_mean,scale=v_std,size=(n,self.dim))
        
        #Uniform distribution with lower bound v_mean and higher bound v_std
        if dist=="uniform":
            self.v_mean=v_mean
            self.v_std=v_std
            _vel=np.random.uniform(low=v_mean,high=v_std,size=(n,self.dim))
        
        #All velocities equal to v_mean
        if dist=="equal":
            self.v_mean=v_mean
            self.v_std=v_std
            _vel=v_mean*np.ones((n,self.dim))
        
        #Randomly switch velocities to negative with probability 0.5
        for i in range(_vel.shape[0]):
            for j in range(_vel.shape[1]):
                if np.random.uniform() > 0.5:
                    _vel[i,j]=-_vel[i,j]
        
        #Store velocities in temporary variable
        self._velocities=_vel
 
    def add_molecules(self,molecule,n,v_mean,v_std,pos_dist="uniform",v_dist="normal",color="black"):
        #Generate initial positions and velocities
        self._generate_initial_positions(n,dist=pos_dist)
        self._generate_initial_velocities(n,v_mean,v_std,dist=v_dist)
        
        #Initialize objects of class Molecule in a list (set mass to 1 as default)
        _add_list=[Molecule(molecule,position=self._positions[i,:],velocity=self._velocities[i,:],color=color,mass=1) for i in range(n)]
        self.molecules.extend(_add_list)
        self.n_molecules+=len(_add_list)
        
        #Calculate KE sum
        self.KE_sum=0
        for i,m in enumerate(self.molecules):
            self.KE_sum+=0.5*m.get_mass()*(np.linalg.norm(m.get_velocity()))**2

        
    def make_matrices(self):
        #Make empty matrices to store positions, velocities, colors, and masses
        self.positions=np.zeros((self.n_molecules,self.dim))
        self.velocities=np.zeros((self.n_molecules,self.dim))
        self.colors=np.zeros(self.n_molecules,dtype="object")
        self.masses=np.zeros(self.n_molecules)
        
        #Iterate over molecules, get their properties and assign to matrices
        for i,m in enumerate(self.molecules):
            self.positions[i,:]=m.get_position()
            self.velocities[i,:]=m.get_velocity()
            self.colors[i]=m.get_color()
            self.masses[i]=m.get_mass()
        
        #Make vectors with magnitudes of positions and velocities
        self.positions_norm=np.linalg.norm(self.positions,axis=1)
        self.velocities_norm=np.linalg.norm(self.velocities,axis=1)
        
        #Make distance matrix
        self.distance_matrix=np.zeros((self.n_molecules,self.n_molecules))
        for i in range(self.distance_matrix.shape[0]):
            for j in range(self.distance_matrix.shape[1]):
                self.distance_matrix[i,j]=np.linalg.norm(self.positions[i,:]-self.positions[j,:])
                
        #Set diagonal entries (distance with itself) to a high value
        #to prevent them for appearing in the subsequent distance filter
        np.fill_diagonal(self.distance_matrix,1e5)
               
    
    def make_velocity_matrix(self):
        self.velocities=np.zeros((self.n_molecules,self.dim))
        for i,m in enumerate(self.molecules):
            self.velocities[i,:]=m.get_velocity()
        
        self.velocities_norm=np.linalg.norm(self.velocities,axis=1)
    
    def update_positions(self):
        #1: Check molecule collisions
        
        #Find molecule pairs that will collide
        collision_pairs=np.argwhere(self.distance_matrix < 2*self.particle_radius)
        
        #If collision pairs exist 
        if len(collision_pairs):
        
            #Go through pairs and remove repeats of indices
            #(for eg., only consider (1,2), remove (2,1))
            pair_list=[]
            for pair in collision_pairs:
                add_pair=True
                for p in pair_list:
                    if set(p)==set(pair):
                        add_pair=False
                        break
                if add_pair:
                    pair_list.append(pair)

            #For every remaining pair, get the molecules, positions, and velocities
            for pair in pair_list:
                m_1=self.molecules[pair[0]]
                m_2=self.molecules[pair[1]]
                
                x_1=m_1.get_position()
                x_2=m_2.get_position()
                
                u_1=m_1.get_velocity()
                u_2=m_2.get_velocity()
                
                #Check if molecules are approaching or departing
                approach_sign=np.sign(np.dot(u_1-u_2,x_2-x_1))
                #If molecules are approaching
                if approach_sign == 1:
                    #Get masses
                    ms_1=m_1.get_mass()
                    ms_2=m_2.get_mass()
                    
                    #Calculate final velocities
                    v_1=u_1 - 2*ms_2/(ms_1 + ms_2) * (np.dot(u_1-u_2,x_1-x_2)/np.linalg.norm(x_1-x_2)**2) * (x_1 - x_2)
                    v_2=u_2 - 2*ms_1/(ms_1 + ms_2) * (np.dot(u_2-u_1,x_2-x_1)/np.linalg.norm(x_2-x_1)**2) * (x_2 - x_1)
                    
                    #Update velocities of the molecule objects
                    m_1.set_velocity(v_1)
                    m_2.set_velocity(v_2)
        
        #2: Update positions
        
        #Iterate over all the molecule objects
        for i,m in enumerate(self.molecules):
            #Get the position, velocity, and mass
            _x=m.get_position()
            _v=m.get_velocity()
            _m=m.get_mass()
            
            #Calculate new position
            _x_new=_x + _v * self.t_step
            
            #3: Check wall collisions
            
            #Check collisions with the top and right walls
            _wall_diff=_x_new - np.array(self.box_dim)           
            #If wall collisions present
            if _wall_diff[_wall_diff>=0].shape[0] > 0:
                #Increment collision counter
                self.wall_collisions+=1
                #Check whether collision in x or y direction
                _coll_ind=np.argwhere(_wall_diff>0)
                #For component(s) to be reflected
                for c in _coll_ind:
                    #Reflect velocity
                    _v[c]=-_v[c]
                    #Increment wall momentum
                    self.wall_momentum+=2*_m*np.abs(_v[c])
                #Update velocity
                m.set_velocity(_v)
                #Update position based on new velocity
                _x_new=_x + _v * self.t_step
            
            #Check collisions with the bottom and left walls    
            if _x_new[_x_new<=0].shape[0] > 0:
                #Increment collision counter
                self.wall_collisions+=1
                #Check whether collision in x or y direction
                _coll_ind=np.argwhere(_x_new<0)
                #For component(s) to be reflected
                for c in _coll_ind:
                    #Reflect velocity
                    _v[c]=-_v[c]
                    #Increment wall momentum
                    self.wall_momentum+=2*_m*np.abs(_v[c])
                #Update velocity    
                m.set_velocity(_v)
                #Update position based on new velocity
                _x_new=_x + _v * self.t_step
                
            #Update position of the molecule object            
            m.set_position(_x_new)
        
        #Construct matrices with updated positions and velocities
        self.make_matrices()
             
    def create_2D_box(self):
        fig=plt.figure(figsize=(10,10*self.box_dim[1]/self.box_dim[0]),dpi=300)
        return fig
        
    def show_molecules(self,i):
        #Clear axes
        plt.cla()
        
        #Plot a line showing the trajectory of a single molecule
        plt.plot(self.x_dynamics[0,0,:i+1],self.x_dynamics[0,1,:i+1],color="red",linewidth=1.,linestyle="-")
        
        #Plot a single molecule in red that is being tracked 
        plt.scatter(self.x_dynamics[0,0,i],self.x_dynamics[0,1,i],color="red",s=20)
        
        #Plot the rest of the molecules
        plt.scatter(self.x_dynamics[1:,0,i],self.x_dynamics[1:,1,i],color=self.colors[1:],s=20)
        
        #Remove ticks on the plot
        plt.xticks([])
        plt.yticks([])
        
        #Set margins to 0
        plt.margins(0)
        
        #Set the limits of the box according to the box dimensions
        plt.xlim([0,self.box_dim[0]])
        plt.ylim([0,self.box_dim[1]])
    
    def safe_division(self,n,d):
        if d==0:
            return 0
        else:
            return n/d
    
    def run_simulation(self,max_time):
        #Print "Starting simulation"
        print("Starting simulation...")
        
        #Make matrices
        self.make_matrices()
        
        #Calculate number of iterations
        self.max_time=max_time
        self.n_iters=int(np.floor(self.max_time/self.t_step))
        
        #Make tensors to store positions and velocities of all molecules at each timestep f
        self.x_dynamics=np.zeros(((self.n_molecules,self.dim,self.n_iters)))
        self.v_dynamics=np.zeros((self.n_molecules,self.n_iters))
        
        #In each iteration
        for i in range(self.n_iters):                   
            #Save positions and velocities to the defined tensors
            self.x_dynamics[:,:,i]=self.positions
            self.v_dynamics[:,i]=self.velocities_norm
            
            #Calculate rms velocity
            self.v_rms=np.sum(np.sqrt(self.velocities_norm**2))/self.velocities_norm.shape[0]
            
            #Print current iteration information
            _P=self.safe_division(self.wall_momentum,i*self.t_step*np.sum(self.box_dim))
            print("Iteration:{0:d}\tTime:{1:.2E}\tV_RMS:{2:.2E}\tWall Pressure:{3}".format(i,i*self.t_step,self.v_rms,_P))
           
            #Call the update_positions function to handle collisions and update positions
            self.update_positions()
            
        #Caclulate final pressure
        self.P=self.wall_momentum/(self.n_iters*self.t_step*np.sum(self.box_dim))
        print("Average pressure on wall: {0}".format(self.P))
        return self.P

    def make_animation(self,filename="KTG_animation.mp4"):
        #Call the function to create the figure
        fig=self.create_2D_box()
        
        #Create the animation
        anim=FuncAnimation(fig,self.show_molecules,frames=self.n_iters,interval=50,blit=False)
        
        #Save animation as a file
        anim.save(filename,writer="ffmpeg")
            
    def plot_hist(self,i):
        #Clear axes
        plt.cla()
        
        #Make histogram
        plt.hist(self.v_dynamics[:,i],density=True,color="plum",edgecolor="black")
        
        #Define axis limits
        plt.xlim([0,3])
        plt.ylim([0,3])
            
    def make_velocity_histogram_animation(self,filename="KTG_histogram.mp4"):
        #Create empty figure
        fig=plt.figure(figsize=(5,5),dpi=500)
        
        #Create animation
        anim_hist=FuncAnimation(fig,self.plot_hist,frames=self.n_iters,interval=50,blit=False)
        
        #Save animation
        anim_hist.save(filename,writer="ffmpeg")
        
    def write_results(self):
        home=os.getcwd()
        os.makedirs(os.path.join(home,"results"),exist_ok=True)
        with open("results/{0}.txt".format(self.name)) as f:
            print("KE={0:.4f} J".format(self.KE_sum),file=f)
            print("T={0:.4f} K".format(self.T),file=f)
            print("P={0:.4f} Pa".format(self.P),file=f)
            print("V={0:.4f} m3".format(self.V),file=f)
            print("N={0:d}".format(self.n_molecules),file=f)
            print("PV/NT={0:.4f}".format(self.P*self.V/(self.T*self.n_molecules)),file=f)
            print("Time:{0:.4f} s".format(self.max_time),file=f)
            print("Iterations:{0:d}".format(self.n_iters),file=f)
            print("Timestep:{0:.4f} s".format(self.t_step),file=f)

if __name__=="__main__":
    #Create simulation object and define input parameters
    sim=Simulation(name="kinetic_theory_simulation",box_dim=[1.0,1.0],t_step=1e-2,particle_radius=1e-2)
    
    #Add N2 molecules to the box
    sim.add_molecules("N2",n=100,v_mean=1.0,v_std=0.2,v_dist="normal")
    
    #Run the simulation and store the pressure output in P
    P=sim.run_simulation(15)
    
    #Make the box animation
    sim.make_animation()
    
    #Make the histogram animation
    sim.make_velocity_histogram_animation()
        
        
        