#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 22:26:06 2020

@author: jakerabinowitz
"""


import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Problem 2a
def read_sat_data():
    """Read the satellite and sunspot data and convert sunspots to mean number per year.
    
    Returns n_sunspots, n_satellites as numpy arrays.
    """
    
    #Import the data
    sun_df = pd.read_csv('SunspotNumber.dat.txt', 
                     names=['year', 'month', 'day', 'sunspots'],  # Give the names of the columns
                     delim_whitespace=True,  # The default is to use ',' as the delimiter. 
                     na_values=-1,  # Tell pandas that -1 means No Data.
                    )

    sat_df = pd.read_csv('SatelliteReentry.dat.txt',
                     names=['year', 'reentries'],  # Give the names of the columns
                     delim_whitespace=True,  # The default is to use ',' as the delimiter. 
                    )

    #Get the mean by year and make sure the years btw the two data sets match up
    n_sunspots = sun_df.groupby(['year'])['sunspots'].mean()
    n_sunspots = pd.DataFrame(n_sunspots.reset_index()) #Reset index to get year as a col name
    sun_year_data = pd.DataFrame(n_sunspots[n_sunspots['year'].isin(list(sat_df["year"]))])

    #Return numpy arrays
    return sun_year_data["sunspots"].values, sat_df["reentries"].values



def sat_likelihood(data, theta):
    """Compute the likelihood of our data given a set of parameters theta.

    In our case, data is a tuple of numpy arrays: (n_sunspots, n_reentries) and theta is the tuple (a,b).

    The likelihood is a Gaussian, assuming the uncertainty in n_reentries is sqrt(n_reentries).
    And note that since only relative likelihoods for different choices of theta are relevant, 
    you can ignore any constant factors that are independent of theta.
    
    Returns the likelihood for this choice of theta.
    """
    
    #get the uncertainty of reentries 
    sat_unc = np.sqrt(data[1])

    #calculate the likelihood: data[0] = sunspots, data[1] = reentries, theat[0] = a, theta[1] = b
    chisq = np.sum(((data[1] - theta[0] - theta[1] * data[0]) / sat_unc) ** 2)
    likelihood = np.exp(-0.5 * chisq)
    
    return likelihood


# Problem 2a (continued)
class MCMC(object):
    """Class that can run an MCMC chain using the Metropolis Hastings algorithm

    This class is based heavily on the "Trivial Metropolis Hastings" algorithm discussed in lecture.
    If you haven't used classes before, you can think of it as just a way of organizing the variables
    and functions related to the MCMC operation.

    You use it by creating an instance of the class as follows:

        mcmc = MCMC(likelihood, data, theta, step_size)

    The parameters here are:

        likelihood is a function returning the likelihood p(data|theta), which needs to be
            defined outside the class.  The function should take two variables (data, theta) and 
            return a single value p(data | theta).

        data is the input data in whatever form the likelihood function is expecting it.  
            This is fixed over the course of running an MCMC chain.

        theta is a list or array with the starting parameter values for the chain.

        step_size is a list or array with the step size in each dimension of theta.


    Then once you have an MCMC object, you can use it by running the following functions:

        mcmc.burn(nburn) runs the chain for nburn steps, but doesn't save the values.

        mcmc.run(nsteps) runs the chain for nsteps steps, saving the results.

        mcmc.accept_fraction() returns what fraction of the candidate steps were taken.

        mcmc.get_samples() returns the sampled theta values as a 2d numpy array.


    There are also simple two plotting functions that you can use to look at the behavior of the chain.

        mcmc.plot_hist() plots a histogram of the sample values for each paramter.  As the chain
            runs for more steps, this should get smoother.
        
        mcmc.plot_samples() plots the sample values over the course of the chain.  If the burn in is
            too short, it should be evident as a feature at the start of these plots.


    Finally, there is only one method you need to write yourself.
    
        mcmc.step() takes a single step of the chain.
    """
    def __init__(self, likelihood, data, theta, step_size, names=None, seed=314159):
        self.likelihood = likelihood
        self.data = data
        self.theta = np.array(theta)
        self.nparams = len(theta)
        self.step_size = np.array(step_size)
        self.rng = np.random.RandomState(seed)
        self.naccept = 0
        self.current_like = likelihood(self.data, self.theta)
        self.samples = []
        if names is None:
            names = ["Paramter {:d}".format(k+1) for k in range(self.nparams)]
        self.names = names            


    def step(self, save=True):
        """Take a single step in the chain"""
        
        #get a theta_new from Normal Dist centered on 0
        theta_new = self.theta + self.rng.normal(0, self.step_size)
    
        #self.theta is theta_old
        ratio = sat_likelihood(self.data, theta_new) / sat_likelihood(self.data, self.theta)
           
        #Decide whether or not to keep theta_new
        if ratio >= 1:
            self.theta = theta_new 
            self.current_like = self.likelihood(self.data, self.theta)
            self.naccept += 1 
            
        else: 
            U = self.rng.uniform()
            
            if U < ratio:
                self.theta = theta_new 
                self.current_like = self.likelihood(self.data, self.theta)
                self.naccept += 1 
                
        if save == True:
            self.samples.append(self.theta)
                         
        
    def burn(self, nburn):
        """Take nburn steps, but don't save the results"""
        for i in range(nburn):
            self.step(save=False)

    def run(self, nsteps):
        """Take nsteps steps"""
        for i in range(nsteps):
            self.step()

    def accept_fraction(self):
        """Returns the fraction of candidate steps that were accpeted so far."""
        if len(self.samples) > 0:
            return float(self.naccept) / len(self.samples)
        else:
            return 0.
        
    def clear(self, step_size=None, theta=None):
        """Clear the list of stored samples from any runs so far.
        
        You can also change the step_size to a new value at this time by giving a step_size as an
        optional parameter value.
        
        In addition, you can reset theta to a new starting value if theta is not None.
        """
        if step_size is not None:
            assert len(step_size) == self.nparams
            self.step_size = np.array(step_size)
        if theta is not None:
            assert len(theta) == self.nparams
            self.theta = np.array(theta)
            self.current_like = self.likelihood(self.data, self.theta)
        self.samples = []
        self.naccept = 0
        
    def get_samples(self):
        """Return the sampled theta values at each step in the chain as a 2d numpy array."""
        return np.array(self.samples)
        
    def plot_hist(self):
        """Plot a histogram of the sample values for each parameter in the theta vector."""
        all_samples = self.get_samples()
        for k in range(self.nparams):
            theta_k = all_samples[:,k]
            plt.hist(theta_k, bins=100)
            plt.xlabel(self.names[k])
            plt.ylabel("N Samples")
            plt.show()
        
    def plot_samples(self):
        """Plot the sample values over the course of the chain so far."""
        all_samples = self.get_samples()
        for k in range(self.nparams):
            theta_k = all_samples[:,k]
            plt.plot(range(len(theta_k)), theta_k)
            plt.xlabel("Step in chain")
            plt.ylabel(self.names[k])
            plt.show()
            
            
sun, sat = read_sat_data()
data = (sun, sat)
theta = (13.11, 0.110)
step_size = (1.1, 0.005)
likelihood = sat_likelihood(data, theta)
  
#Create an MCMC object 
mcmc = MCMC(sat_likelihood, data, theta, step_size)
mcmc.run(int(1e5))
mcmc.get_samples()

ab_array = mcmc.get_samples()

#np.cov(ab_array)

a, b  = zip(*ab_array)
a = np.asarray(a)
b = np.asarray(b)

# Problem 2b
def calculate_mean(mcmc):
    """Calculate the mean of each parameter according to the samples in the MCMC object.

    Returns the mean values as a numpy array.
    """
    
    sample_array = mcmc.get_samples()
    #Seperate the 2D array into an a and b array
    a, b  = zip(*sample_array)
    
    #Turn values into numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)

    return np.array([a.mean(), b.mean()])

calculate_mean(mcmc)

    
def calculate_cov(mcmc):
    """Calculate the covariance matrix of the parameters according to the samples in the MCMC object.

    Returns the covariance matrix as a 2d numpy array.
    """
    sample_array = mcmc.get_samples()
    #Seperate the 2D array into an a and b array
    a, b  = zip(*sample_array)
    
    a = np.asarray(a)
    b = np.asarray(b)

    #Turn to array
    ab_np = np.array([a, b])

    return np.cov(ab_np)

# Problem 2c
def plot_corner(mcmc):
    """Make a corner plot for the parameters a and b with contours corresponding to the same
    delta chisq contours we drew in homework 4.
    """
    # Hint: Use the corner.corner function
    
    # YOUR CODE HERE
    raise NotImplementedError()

import corner
import numpy as np

x = mcmc.get_samples()
fig = corner.corner(x, levels=(.1, .5,))

ax.set_title(r'$\chi^2(a,b)$)')
fig.suptitle("correct `one-sigma' level");


    
