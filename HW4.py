#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 21:21:44 2020

@author: jakerabinowitz
"""

import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt


#Import the data
sunSpots = np.genfromtxt('/Users/jakerabinowitz/Dropbox/Senior Year Fall/PHYS358/SunspotNumber.dat.txt', names=['year','month','day','number'])
data = pd.DataFrame(sunSpots)


sun_df = pd.read_csv("/Users/jakerabinowitz/Dropbox/Senior Year Fall/PHYS358/SunspotNumber.dat.txt", 
                     names=['year', 'month', 'day', 'sunspots'],  # Give the names of the columns
                     delim_whitespace=True,  # The default is to use ',' as the delimiter. 
                     na_values=-1,  # Tell pandas that -1 means No Data.
                    )


sat_df = pd.read_csv('/Users/jakerabinowitz/Dropbox/Senior Year Fall/PHYS358/SatelliteReentry.dat.txt',
                     names=['year', 'reentries'],  # Give the names of the columns
                     delim_whitespace=True,  # The default is to use ',' as the delimiter. 
                    )


n_reentries = pd.DataFrame(sat_df.set_index('year')['reentries'])


# Problem 1a
def mean_sunspots_by_month(df):
    """
    Calculate the mean number of sunspots observed in each month.

    The input pandas data frame include columns named 'year', 'month', 'day', and 'sunspots'.
    
    Rows where the number of sunspots is NaN should be ignored when taking the average.
    
    Returns a pandas Series indexed by (year, month) with the mean number of sunspots in each month.
    """
    
    #drop all the NaN values and save thd df inplace 
    df.dropna(inplace = True)
    groupByMonth_df = pd.DataFrame(df.groupby(["year", "month"])["sunspots"].mean())
    
    return groupByMonth_df.reset_index()


mean_sunspots_by_month(sun_df)

# Problem 1b
def mean_sunspots_by_year(df):
    """
    Calculate the mean number of sunspots observed in each year.

    The input pandas data frame include columns named 'year', 'month', 'day', and 'sunspots'.
    
    Rows where the number of sunspots is NaN should be ignored when taking the average.
    
    Returns a pandas Series indexed by (year) with the mean number of sunspots in each year.
    """
    df.dropna(inplace = True)
    groupByYear_df = pd.DataFrame(df.groupby("year")["sunspots"].mean())
    
    return groupByYear_df.reset_index()
    
mean_sunspots_by_year(sun_df)



# Problem 2a
def fit_reentries_vs_sunspots(n_sunspots, n_reentries):
    """
    Fit a linear regression to the relation
    
        n_reentries = a + b n_sunspots
        
    The inputs are pandas Series instances, indexed by year.  However, they don't (necessarily) have the
    same years, so this function will only consider the subset of years common to both series.
    
    The uncertainty in the number of reentries is taken to be sigma_N = sqrt(N).
    
    Returns a, b
    """
    #Getting the years to match
    n_reentries = pd.DataFrame(n_reentries.reset_index()) #added this line to set 'year' back as a column
    sun_df_year = mean_sunspots_by_year(n_sunspots)
    corr_sun_year_df = pd.DataFrame(sun_df_year[sun_df_year['year'].isin(list(n_reentries["year"]))])
    
    #Get the standard deviation
    sat_df["stdev"] = np.sqrt(sat_df["reentries"])
    
    #initialize the s values
    s = 0
    s_x = 0
    s_y = 0
    s_xx = 0 
    s_xy = 0
    
    #create the sums
    for i in range(sat_df.shape[0]):
        s += 1 / (sat_df["stdev"][i] ** 2)
        s_x += (corr_sun_year_df["sunspots"].iloc[i]) / (sat_df["stdev"][i] ** 2)
        s_y += (sat_df["reentries"][i]) / (sat_df["stdev"][i] ** 2)
        s_xx += (corr_sun_year_df["sunspots"].iloc[i] ** 2) / (sat_df["stdev"][i] ** 2)
        s_xy += (corr_sun_year_df["sunspots"].iloc[i] * sat_df["reentries"][i]) / (sat_df["stdev"][i] ** 2)
    
    #find det, a, b
    det = (s * s_xx) - (s_x ** 2)
    a = (s_xx * s_y - s_x * s_xy) / det
    b = (s * s_xy - s_x * s_y) / det
    
    return a, b

fit_reentries_vs_sunspots(sun_df, sat_df)


# Problem 2b
def calculate_cov(n_sunspots, n_reentries):
    """Calculate the covariance matrix of (a,b), the linear fit to
    
        n_reentries = a + b n_sunspots
    
    Returns 2D numpy array [[ sigma_a^2  sigma_ab  ]
                            [ sigma_ab   sigma_b^2 ]]
    """
    #Getting the years to match
    n_reentries = pd.DataFrame(n_reentries.reset_index()) #added this line to set 'year' back as a column
    sun_df_year = mean_sunspots_by_year(n_sunspots)
    corr_sun_year_df = pd.DataFrame(sun_df_year[sun_df_year['year'].isin(list(n_reentries["year"]))])
    
    #Get the standard deviation
    sat_df["stdev"] = np.sqrt(sat_df["reentries"])
    
    #initialize the s values
    s = 0
    s_x = 0
    s_y = 0
    s_xx = 0 
    s_xy = 0
    
    #create the sums
    for i in range(sat_df.shape[0]):
        s += 1 / (sat_df["stdev"][i] ** 2)
        s_x += (corr_sun_year_df["sunspots"].iloc[i]) / (sat_df["stdev"][i] ** 2)
        s_y += (sat_df["reentries"][i]) / (sat_df["stdev"][i] ** 2)
        s_xx += (corr_sun_year_df["sunspots"].iloc[i] ** 2) / (sat_df["stdev"][i] ** 2)
        s_xy += (corr_sun_year_df["sunspots"].iloc[i] * sat_df["reentries"][i]) / (sat_df["stdev"][i] ** 2)
    
    #find det and cov matrix
    det = (s * s_xx) - (s_x ** 2)
        
    sigma_aa = s_xx / det
    sigma_bb = s / det
    sigma_ab = -s_x / det # this is the covariance 
    
    cov_matrix = np.array([[sigma_aa, sigma_ab], [sigma_ab, sigma_bb]])
    
    return cov_matrix

calculate_cov(sun_df, sat_df)

a = fit_reentries_vs_sunspots(sun_df, sat_df)[0]
b = fit_reentries_vs_sunspots(sun_df, sat_df)[1]
n_sunspots = sun_df
n_reentries = sat_df
cov = calculate_cov(n_sunspots, n_reentries)

# Problem 2c
def plot_linear_fit(ax, n_sunspots, n_reentries, a, b, cov):
    """Plot the linear fit n_reentries = a + b n_sunspots on the given matplotlib axis.
    """
    n_reentries = pd.DataFrame(n_reentries.reset_index()) #added this line to set 'year' back as a column
    sun_df_year = mean_sunspots_by_year(n_sunspots)
    corr_sun_year_df = pd.DataFrame(sun_df_year[sun_df_year['year'].isin(list(n_reentries["year"]))])
    
    plt.plot(corr_sun_year_df["sunspots"], sat_df["reentries"], ".", color = "red")
    plt.errorbar(corr_sun_year_df["sunspots"], sat_df["reentries"], yerr = sat_df["stdev"], 
             fmt = " ", ecolor = "black", capsize = 3)

    plt.style.use('fivethirtyeight')
    x = np.linspace(0, 300, 250)
    plt.plot(x, b * x + a, '-b', label = "y = 0.1099x + 13.11" )
    plt.grid()
    plt.ylabel("Number of Satellite Reentries")
    plt.xlabel("Average Number of Sunspots per Year")
    plt.title("Number of Satellite Reentries vs Average Number of Sunspots per Year")
    plt.legend(loc = "upper left")
    plt.show()
    
fig, ax = plt.subplots(1,1, figsize=(14,8))
plot_linear_fit(ax, n_sunspots, n_reentries, a, b, cov)
plt.show()
    


#These a and b value minimize the chisq value
a = fit_reentries_vs_sunspots(sun_df, sat_df)[0]
b = fit_reentries_vs_sunspots(sun_df, sat_df)[1]

# Problem 2d
# First a helper function to calculate the chisq value for any proposed values of a,b.
def calculate_chisq(n_sunspots, n_reentries, a, b):
    """Calculate chisq for a given proposed solution (a, b), assumuing var(N) = N.

    chisq = Sum_i (Nr_i - N_fit(Ns_i))^2 / var(Nr_i),
    where N_fit(Ns) = a + b Ns
    
    Returns chisq for the proposed values of a and b.
    """ 
    n_reentries = pd.DataFrame(n_reentries.reset_index()) #added this line to set 'year' back as a column
    sun_df_year = mean_sunspots_by_year(n_sunspots)
    corr_sun_year_df = pd.DataFrame(sun_df_year[sun_df_year['year'].isin(list(n_reentries["year"]))])

    chisq = 0 
    #loop to calculate chisq
    for i in range(sat_df.shape[0]):
        chisq += ((sat_df["reentries"][i] - a - b * corr_sun_year_df["sunspots"].iloc[i]) / (sat_df["stdev"][i])) ** 2
        
    return chisq
    
calculate_chisq(sun_df, sat_df, a, b)  


# Problem 2d (continued)
# Now we can use the above function to plot chisq values for many a,b values as a contour plot.
def plot_chisq_ellipses(ax, n_sunspots, n_reentries):
    """Plot ellipses at the 1, 2, and 3-sigma contours of a,b
    """
    
    a, b = fit_reentries_vs_sunspots(n_sunspots, n_reentries)

    cov = calculate_cov(n_sunspots, n_reentries)
    sigma_aa = cov[0,0]
    sigma_bb = cov[1,1]
    
    xMesh = np.linspace(a - 4 * np.sqrt(sigma_aa), a + 4 * np.sqrt(sigma_aa), 1000)
    yMesh = np.linspace(b - 4 * np.sqrt(sigma_bb), b + 4 * np.sqrt(sigma_bb), 1000)

    x, y = np.meshgrid(xMesh, yMesh)
    chisq = calculate_chisq(n_sunspots, n_reentries, x, y)
    
    chisq_ab = calculate_chisq(n_sunspots, n_reentries, a, b)
    ax.contourf(x, y, chisq, levels = [chisq_ab + 2.30, chisq_ab + 6.17, chisq_ab + 11.8])
    plt.ylabel("b parameter")
    plt.xlabel("a parameter")
    plt.title("Contour Map Showing 1, 2, 3 Sigma Confidence Intervals")
    plt.grid()
    plt.plot(a, b, 'o', color = "black")


fig, ax = plt.subplots(1,1, figsize=(14,8))
plot_chisq_ellipses(ax, n_sunspots, n_reentries)
plt.show()
    

N = 90
xmin = 18
xmax = 38
true_a = 200
true_b = 10
sigma = 10


xList = np.random.uniform(xmin, xmax, N)
xList.sort()

#initialize a yList
yList = []

#fill the yList 
for x in xList:
    print(x)
    y = true_a + true_b * x
    yList.append(y)

#take N random normal distribution y values
yGauss = []
for y in yList:
    print(y)
    yG = np.random.normal(y, sigma)
    yGauss.append(yG)
    

# Problem 3a
def sim_linear(N, xmin, xmax, a, b, sigma):
    """Simulate an experiment with the true relation y = a + b x.
    
    N is the number of data points in the returned arrays.
    xmin, xmax give the range of x values.  The returned x values are uniformly sampled from [xmin, xmax]
    a, b are the true coefficients of the linear relation y = a + b x
    sigma is a constant Gaussian uncertainty of the y values relative to the truth values.

    Returns x, y, both N-element numpy arrays
    """
    
    #Get N values for x in range xmin to xmax
    xList = np.random.uniform(xmin, xmax, N)
    
    xList.sort()
    #initialize a yList
    yList = []

    #fill the yList 
    for x in xList:
        y = true_a + true_b * x
        yList.append(y)

    #take N random normal distribution y values
    yGauss = []
    for y in yList:

        yG = np.random.normal(y, sigma)
        yGauss.append(yG)
    
    return xList, yGauss



x = sim_linear(90, 18, 38, 200, 10, 10)[0]
y = sim_linear(90, 18, 38, 200, 10, 10)[1]

plt.plot(x, y, ".")
plt.show()

def fit_linear(x, y, sigma):
    """Perform a linear fit y = a + b x, given a constant uncertainty, sigma, of the y values.
    
    Returns a, b, sigma_a, sigma_b
    """
    #initialize the s values
    s = 0
    s_x = 0
    s_y = 0
    s_xx = 0 
    s_xy = 0
    
    #create the sums
    for i in range(len(x)):
        s += 1 / (sigma ** 2)
        s_x += (x[i]) / (sigma ** 2)
        s_y += (y[i]) / (sigma ** 2)
        s_xx += (x[i] ** 2) / (sigma ** 2)
        s_xy += (x[i] * y[i]) / (sigma ** 2)
    
    #find det, a, b
    det = (s * s_xx) - (s_x ** 2)
    a = (s_xx * s_y - s_x * s_xy) / det
    b = (s * s_xy - s_x * s_y) / det
    sigma_aa = s_xx / det
    sigma_bb = s / det
    
    return a, b, np.sqrt(sigma_aa), np.sqrt(sigma_bb)
    
fit_linear(x, y, 10)
    


N = 90
xmin = 18
xmax = 38
true_a = 200
true_b = 10
sigma = 10
Nsim = 10000


def getChisq(a, b, x, y, N):
    chisq = 0
    for i in range(N):
        chisq += ((y[i] - a - b * x[i]) / (np.sqrt(stat.variance(y)))) ** 2
    
    return chisq

# Problem 3b
def run_simulation(N, xmin, xmax, true_a, true_b, sigma, Nsim):
    '''
    1. Generates Nsim realizations of the dataset.
    2. Determine the MLE of a, b, sigma_a, sigma_b for 
       each realization
    3. Find the chisq of each realization
    4. Calculate the inverse-variance-weighted mean of
       a and b and the corresponding uncertainties
       for each realiation.
    
    Returns: chisqs, mean_a, mean_b, sig_meana, sig_meanb
    '''
    aList = []
    bList = []
    sigma_a = []
    sigma_b = []
    chisqList = []
    for i in range(Nsim):
        x, y = sim_linear(N, xmin, xmax, a, b, sigma)
        a_i = fit_linear(x, y, sigma)[0]
        aList.append(a)
        b_i = fit_linear(x, y, sigma)[1]
        bList.append(b)
        sigma_a.append(fit_linear(x, y, sigma)[2])
        sigma_b.append(fit_linear(x, y, sigma)[3])
        chisqList.append(getChisq(a_i, b_i, x, y, N))

    return chisqList, np.mean(a), np.mean(b), np.mean(sigma_a), np.mean(sigma_b)



chisqs  = run_simulation(N, xmin, xmax, true_a, true_b, sigma, Nsim)[0]

from scipy.stats import chi2

# Problem 3c
def plot_chisq_hist(ax, chisqs, nu):
    '''Plots a histogram of chisq values
    Also overplots a chisq distribution for nu degrees of freedom.
    '''
    n, bins, patches = plt.hist(chisqs, density=False, facecolor='g')
    plt.ylabel("Number of Occurrences")
    plt.xlabel("Chisq Value")
    x = np.linspace(0, 6, 100)
    plt.plot(x, chi2.pdf(x, nu)* 100000,'r-', lw=5, alpha=0.6, label='chi2 pdf')


fig, ax = plt.subplots(1, 1, figsize=(14,8))
plot_chisq_hist(ax, chisqs, N-2)
plt.show()



























