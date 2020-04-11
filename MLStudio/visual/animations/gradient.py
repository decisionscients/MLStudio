#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : gradient.py                                                       #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, April 10th 2020, 3:53:31 pm                         #
# Last Modified : Friday, April 10th 2020, 3:53:31 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Visualization classes for diagnostics and evaluation"""
# --------------------------------------------------------------------------- #
import datetime
from IPython.display import HTML
import itertools
from itertools import zip_longest
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib import animation, rc, rcParams
rcParams['animation.embed_limit'] = 50

from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from numpy import array, newaxis
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from mlstudio.utils.data_manager import todf
from mlstudio.utils.file_manager import save_fig, save_csv, save_gif


# --------------------------------------------------------------------------- #
#                             SingleModelSearch3D                             #  
# --------------------------------------------------------------------------- #
class SingleModelSearch3D:
    """Animates search for 2D optimization in 3D"""    

    def __init__(self):
        pass        

    def _cost_mesh(self,X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))        

    def search(self, model, directory=None, filename=None, fontsize=None,
                    interval=200, secs=10, maxframes=100):
        '''Animates gradient descent for 2D problems in 3D.
        '''        

        # Unpack model parameters
        params = model.get_params()   

        # Designate plot area
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sns.set(style="whitegrid", font_scale=1)

        # Create index for n <= maxframes number of points
        iterations = np.arange(1, model.history_.total_epochs+1)
        idx = np.arange(0,model.history_.total_epochs)
        nth = math.floor(model.history_.total_epochs/maxframes)
        nth = max(nth,1) 
        idx = idx[::nth]
        points = len(idx)

        # Create the x=theta0, y=theta1 grid for plotting
        weights = todf(model.history_.epoch_log['theta'], stub='theta_')        
        theta0 = weights['theta_0']
        theta1 = weights['theta_1']

        # Format frames per second
        fps = math.floor(points/secs) if points >= secs else math.floor(secs/points)      

        # Establish boundaries of plot
        theta0_min = min(theta0)
        theta1_min = min(theta1)
        theta0_max = max(theta0) + abs(max(theta0)-min(theta0))
        theta1_max = max(theta1) + abs(max(theta1)-min(theta1))
        theta0_mesh = np.linspace(theta0_min, theta0_max, 50)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 50)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

        # Create cost grid based upon X,y and the grid of thetas
        Js = np.array([self._cost_mesh(model.X, model.y, THETA)
                    for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
        Js = Js.reshape(theta0_mesh.shape)

        # Set Title
        plt.rc('text', usetex=True)
        title = model.name + '\n' + r'$\alpha$' + " = " + str(round(params['learning_rate'],3))
        ax.set_title(title, color='k', pad=30, fontsize=20)                            
        if fontsize:            
            display = ax.text2D(0.1,0.92, '', transform=ax.transAxes, color='k', fontsize=fontsize)
        else:            
            display = ax.text2D(0.2,0.92, '', transform=ax.transAxes, color='k')             
        # Set face, tick,and label colors 
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # Make surface plot
        ax.plot_surface(theta0_mesh, theta1_mesh, Js, rstride=1,
                cstride=1, cmap='jet', alpha=0.5, linewidth=0)
        ax.set_xlabel(r'Intercept($\theta_0$)')
        ax.set_ylabel(r'Slope($\theta_1$)')
        ax.set_zlabel('Cost')        
        ax.view_init(elev=30., azim=30)

        # Build the empty line plot at the initiation of the animation
        line2d, = ax.plot([], [], [], 'b-', lw = 1.5)
        line3d, = ax.plot([], [], [], 'r-', lw = 1.5)        
        point2d, = ax.plot([], [], [], 'bo')
        point3d, = ax.plot([], [], [], 'ro')

        def init():

            # Initialize 3d line and point
            line3d.set_data([],[])
            line3d.set_3d_properties([])
            point3d.set_data([], [])
            point3d.set_3d_properties([])

            # Initialize 2d line and point
            line2d.set_data([],[])
            line2d.set_3d_properties([])
            point2d.set_data([], [])
            point2d.set_3d_properties([])            

            # Initialize display
            display.set_text('')
            return (line2d, point2d, line3d, point3d, display,)

        # Animate the optimization path as it converges
        def animate(i):
            # Animate 3d Line
            line3d.set_data(theta0[:idx[i]], theta1[:idx[i]])
            line3d.set_3d_properties(model.history_.epoch_log['train_cost'][:idx[i]])
            # Animate 3d points
            point3d.set_data(theta0[idx[i]], theta1[idx[i]])
            point3d.set_3d_properties(model.history_.epoch_log['train_cost'][idx[i]])
            # Animate 2d Line
            line2d.set_data(theta0[:idx[i]], theta1[:idx[i]])
            line2d.set_3d_properties(0)
            # Animate 2d points
            point2d.set_data(theta0[idx[i]], theta1[idx[i]])
            point2d.set_3d_properties(0)            
            # Rotate
            ax.view_init(elev=30., azim=i*.30)
            # Update text
            metrics = 'Iteration: '+ str(iterations[idx[i]]) + \
                      r'$\quad\theta_0=$ ' + str(round(theta0[idx[i]],3)) \
                          + r'$\quad\theta_1=$ ' + str(round(theta1[idx[i]],3)) +\
                            '  Cost: ' + \
                                str(np.round(model.history_.epoch_log['train_cost'][idx[i]], 3))
            display.set_text(metrics)            

            return(line3d, point3d, line2d, point2d, display)

        # create animation using the animate() function
        surface_ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(idx),
                                            interval=interval, blit=True, repeat_delay=3000)
        if directory is not None:
            if filename is None:
                filename = model.algorithm + ' Search Path Learning Rate ' + str(params['learning_rate']) +  '.gif'
            save_gif(surface_ani, directory, filename, fps)
        return(surface_ani)

# --------------------------------------------------------------------------- #
#                              SingleModelFit2D                               #   
# --------------------------------------------------------------------------- #
class SingleModelFit2D:
    """Animates gradient descent fit in 2D for 2D optimation problem."""

    def fit(self, model, directory=None, filename=None, fontsize=None,
                 interval=200, secs=10, maxframes=100):

        # Unpack model results
        params = model.get_params()         
        X = model.X
        y = model.y        

        # Create index for n <= maxframes number of points
        iterations = np.arange(1, model.history_.total_epochs+1)
        idx = np.arange(0,model.history_.total_epochs)
        nth = math.floor(model.history_.total_epochs/maxframes)
        nth = max(nth,1) 
        idx = idx[::nth]
        points = len(idx)

        # Extract data for plotting
        x = model.X[:,1]        
        cost = model.history_.epoch_log['train_cost']        
        weights = todf(model.history_.epoch_log['theta'], stub='theta_')        
        theta0 = weights['theta_0']
        theta1 = weights['theta_1']
        theta = np.array([theta0, theta1])

        # Format frames per second
        fps = math.floor(points/secs) if points >= secs else math.floor(secs/points)      
        
        # Render scatterplot
        fig, ax = plt.subplots(figsize=(12,8))
        sns.set(style="whitegrid", font_scale=1)
        sns.scatterplot(x=x, y=y, ax=ax)
        ax.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
        # Set face, tick,and label colors 
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        # Initialize line
        line, = ax.plot([],[],'r-', lw=2)
        # Set Title, Annotations and label
        title = model.algorithm + '\n' + r'$\alpha$' + " = " + str(round(params['learning_rate'],3))
        ax.set_title(title, color='k', fontsize=20)
        if fontsize:            
            display = ax.text(0.1, 0.9, '', transform=ax.transAxes, color='k', fontsize=fontsize)
        else:            
            display = ax.text(0.2, 0.9, '', transform=ax.transAxes, color='k')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        fig.tight_layout()

        # Build the empty line plot at the initiation of the animation
        def init():
            line.set_data([],[])
            display.set_text('')
            return (line, display,)

        # Animate the regression line as it converges
        def animate(i):

            # Animate Line
            y=X.dot(theta[:,idx[i]])
            line.set_data(x,y)

            # Animate text
            display.set_text('Iteration: '+ str(iterations[idx[i]]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[idx[i]],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[idx[i]],3)) +
                            ' Cost: ' + str(round(cost[idx[i]], 3)))
            return (line, display)

        # create animation using the animate() function
        line_gd = animation.FuncAnimation(fig, animate, init_func=init, frames=len(idx),
                                            interval=interval, blit=True, repeat_delay=3000)
        if directory is not None:
            if filename is None:
                filename = model.algorithm + ' Fit Plot Learning Rate ' + str(params['learning_rate']) +  '.gif'
            save_gif(line_gd, directory, filename, fps)
        plt.close(fig)  
        return(line_gd)

# --------------------------------------------------------------------------- #
#                            MultiModelSearch3D                          #  
# --------------------------------------------------------------------------- #
class MultiModelSearch3D(animation.FuncAnimation):
    """Animates gradient descent in 3D for multiple 2D optimization problems."""

    def __init__(self):
        pass

    def _anim8(self, *paths, zpaths, methods=[], frames=100, 
                 interval=60, repeat_delay=5, blit=False, **kwargs):

        self.paths = paths
        self.zpaths = zpaths

        if frames is None:
            frames = max(path.shape[1] for path in paths)
            self.nth = [1 for p in paths]
        else:
            self.nth = [max(1,math.floor(path.shape[1] / frames)) for path in paths]
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0,1,len(paths)))
        self.lines = [self.ax.plot([], [], [], alpha=0.7, label=method, c=c, lw=2)[0] 
                      for _, method, c in zip_longest(paths, methods, colors)]

        ani = animation.FuncAnimation(self.fig, self._update, init_func=self._init,
                                      frames=frames, interval=interval, blit=blit,
                                      repeat_delay=repeat_delay, **kwargs)
        self.ax.legend(loc='upper left')                                      
        return ani

    def _init(self):
        for line in self.lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return self.lines

    def _update(self, i):
        for line, nth, path, zpath in zip(self.lines, self.nth, self.paths, self.zpaths):
            self.ax.view_init(elev=60., azim=i*.30)
            line.set_data(*path[::,:i*nth])
            line.set_3d_properties(zpath[:i*nth])
        return self.lines

    def _cost_mesh(self,X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y))) 

    def _meshgrid(self, models):
        theta0_mins = []
        theta1_mins = []
        theta0_maxs = []
        theta1_maxs = []

        for _, v in models.items():
            theta0_mins.append(min(item[0] for item in v.history_.batch_log.get('theta')))
            theta1_mins.append(min(item[1] for item in v.history_.batch_log.get('theta')))
            theta0_maxs.append(max(item[0] for item in v.history_.batch_log.get('theta')))
            theta1_maxs.append(max(item[1] for item in v.history_.batch_log.get('theta')))
        
        theta0_min = min(theta0_mins)
        theta1_min = min(theta1_mins)
        theta0_max = max(theta0_maxs)
        theta1_max = max(theta1_maxs)

        theta0_mesh = np.linspace(theta0_min, theta0_max, 100)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 100)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)
        return theta0_mesh, theta1_mesh

    def _surface(self, X, y, models, height, width):

        # Designate plot area
        self.fig = plt.figure(figsize=(12*width, 8*height))
        self.ax = self.fig.add_subplot(111, projection='3d')
        sns.set(style="whitegrid", font_scale=1)        

        # Create cost grid based upon x,y the grid of thetas
        theta0_mesh, theta1_mesh = self._meshgrid(models)
        Js = np.array([self._cost_mesh(X, y, THETA) 
                    for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
        Js = Js.reshape(theta0_mesh.shape)

        # Set Title
        title = 'Gradient Descent Trajectories'
        self.ax.set_title(title, color='k', pad=30, fontsize=20)                       

        # Set face, tick,and label colors 
        self.ax.set_facecolor('w')
        self.ax.tick_params(colors='k')
        self.ax.xaxis.label.set_color('k')
        self.ax.yaxis.label.set_color('k')
        # make the panes transparent
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        self.ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        self.ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        self.ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # Make surface plot
        self.ax.plot_surface(theta0_mesh, theta1_mesh, Js, rstride=1,
                cstride=1, cmap='jet', alpha=0.5, linewidth=0)
        self.ax.set_xlabel(r'Bias($\theta_0$)')
        self.ax.set_ylabel(r'Coefficient($\theta_1$)')
        self.ax.set_zlabel('Cost')        
        self.ax.view_init(elev=30., azim=30)        

    def _get_data(self, models):
        
        paths=[]
        zpaths=[]
        methods = []        
        for k, v in models.items():    
            paths.append(np.array(v.history_.epoch_log.get('theta')).T)
            zpaths.append(np.array(v.history_.epoch_log.get('train_cost'))) 
            methods.append(k)  
            X = v.X_design_
            y = v.y_
        return(X, y, paths, zpaths, methods)

    def search(self, models, frames=100, directory=None, filename=None, fps=5,
               height=1, width=1):

        X, y, paths, zpaths, methods = self._get_data(models)        
        self._surface(X, y, models, height, width)        
        ani = self._anim8(*paths, zpaths=zpaths, methods=methods, frames=frames)
        if directory is not None:
            if filename is None:
                filename = 'Gradient Descent Trajectories.gif'
            save_gif(ani, directory, filename, fps)
        return ani


# --------------------------------------------------------------------------- #
#                           GRADIENT FIT 2D CLASS                             #   
# --------------------------------------------------------------------------- #
class MultiModelFit2D(animation.FuncAnimation):
    """Animates fit for multiple 2D optimizations.""" 

    def __init__(self):
        pass

    def _anim8(self, *paths, methods=[], frames=None, 
                 interval=60, repeat_delay=5, blit=False, **kwargs):

        self.paths = paths

        if frames is None:
            frames = max(path.shape[1] for path in paths)
            self.nth = [1 for p in paths]
        else:
            self.nth = [max(1,math.floor(path.shape[1] / frames)) for path in paths]
        cmap = plt.get_cmap('jet')
        colors = cmap(np.linspace(0,1,len(paths)))
        self.lines = [self.ax.plot([], [], label=method, c=c, lw=2)[0] 
                      for _, method, c in zip_longest(paths, methods, colors)]

        ani = animation.FuncAnimation(self.fig, self._update, init_func=self._init,
                                      frames=frames, interval=interval, blit=blit,
                                      repeat_delay=repeat_delay, **kwargs)
        self.ax.legend(loc='upper left')                                      
        return(ani)                                                  

    def _init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def _update(self, i):
        for line, nth, path, in zip(self.lines, self.nth, self.paths):
            y = self.X.dot(path[::,i*nth])
            line.set_data(self.X[:,1], y)            
        return self.lines

    def _scatterplot(self, x_label=None, y_label=None):

        # Designate plot area
        self.fig, self.ax = plt.subplots(figsize=(12,8))
        sns.set(style="whitegrid", font_scale=1)
        sns.scatterplot(x=self.X[:,1], y=self.y, ax=self.ax)

        # Set face, tick,and label colors 
        self.ax.set_facecolor('w')
        self.ax.tick_params(colors='k')
        self.ax.xaxis.label.set_color('k')
        self.ax.yaxis.label.set_color('k')

        # Set Title, Annotations and label
        title = 'Gradient Descent Model Fit'
        self.ax.set_title(title, color='k', fontsize=20)
        self.ax.set_xlabel(x_label) if x_label is not None else self.ax.set_xlabel("X")
        self.ax.set_ylabel(y_label) if x_label is not None else self.ax.set_ylabel("y")
        self.fig.tight_layout()      

    def _get_data(self, models):
        
        paths=[]
        methods = []
        for k, v in models.items():    
            paths.append(np.array(v.history_.batch_log.get('theta')).T)
            methods.append(k)  
            self.X = v.X_design_
            self.y = v.y_

        return paths, methods
    def fit(self, models, frames=100, x_label=None, y_label=None, 
            directory=None, filename=None, fps=5):

        paths, methods = self._get_data(models)
        self._scatterplot(x_label, y_label)
        ani = self._anim8(*paths, methods=methods, frames=frames)
        if directory is not None:
            if filename is None:
                filename = 'Gradient Descent Fit.gif'            
            save_gif(ani, directory, filename, fps)
        return(ani)        