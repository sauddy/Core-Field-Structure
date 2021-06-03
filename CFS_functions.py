
# coding: utf-8

# In[1]:

## modified : 23 April 2018 
# the fucntion now can read the error from the Plummer fit.
# additionally i have added a 10 percent error on the estimate of sigma_NT_C
## Functions for the field analysis##

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
from scipy import optimize
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy import integrate
import function as mf


# In[2]:

# parameters that are needed.
distinpc = 120.0
pc = 3.0857e+18 # cm 
AU = 1.496e+13  # cm
amu = 1.6605402e-24 # g
mu = 2.33*amu
mperH2= 2.8* amu
Msun=1.9891e+33 # g
G = 6.67259e-8 # cm3 g-1 s-2
# 1 Gauss = g^(1/2) * cm^{-1/2}* s^-1
##


# In[3]:

## read the columndensity data file:
def readcolumndensity(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    radius_list = []        # units arcsec 
    ringcolumn_list= []      # units cm^{-3}
    dringcolumn_list = []

    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        radius_list.append(float(t[0]))
        ringcolumn_list.append(float(t[1]))
        dringcolumn_list.append(float(t[2])) 

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    radius= np.asarray(radius_list)
    columndensity = np.asarray(ringcolumn_list)
    columndensityerror =np.asarray(dringcolumn_list)
    return (radius, columndensity,columndensityerror)


# In[4]:

def plummerfunction(radius,sigma_flat,r_fl,eta):
    sigma = sigma_flat / (1+(radius/r_fl)**2)**eta
    return sigma


# In[5]:

def plummerparameter(sigma_flat,rf,eta):
    index = eta + 0.5
#     print(index)
    r_flat = rf*distinpc*AU
    r_flatinpc = r_flat/pc 

    # Calculation of n_flat
    # we need to evaluate the parameter Ap (refer arzoumanian et al 2011)

    def Ap(u):
        return 1/ (1+u**2)**index
    def bounds_u():
        return [-2500, 2500]

    result = integrate.nquad(Ap, [bounds_u])
#     print(result)
    n_flat=(sigma_flat)/result[0]/r_flat
#     print(n_flat)
    return(n_flat, r_flatinpc,index)


# In[ ]:

## Errors in magnetic field using monte carlo##

def B_sigmaerror(index,r_flatinpc, n_flat,Plummermodel_error, sigma_NT_C,rc_,r, k):
    '''
    This function estimate the disribution of B field at a particular radius
    Input Parameter : index 
                    : r_flatinpc
                    : n_flat
                    : sigma_NT_c
                    :The fit error of the Plummer Model (added on 23/04/18)
                    : radius at which the field is estiamtes
                    : k the \kappa for the flux freezing model
                    
    last modified on 30 July 2018 
    Added the error on rc and beta
    for rc the error variation is 10 percent 
    for beta the error can vary between 0.5 - 0.8
    
    '''
    
    B_sigma = []
    B_mean = []
    B_1quartile = []
    B_3quartile = []
    npts = 1000
    nsim = 1000
    beta = 0.5
    ## the error were added on 23 April 2018
    eta_err = Plummermodel_error[0]
    r_0_err = Plummermodel_error[1]
    n_0_err = Plummermodel_error[2]
    ## the theory at each radius for each simulation we get a B distribution and we estimate the mean, 1st quartile and 
    ## 3rd quartile and B_sigma. Then we run 1000 simulations and take the mean of the above quantity. 
    for i in range (1, nsim):
        power= index + np.random.standard_normal((npts,))*eta_err
        r_0 = r_flatinpc  + np.random.standard_normal((npts,))*r_0_err
        n_0 = n_flat + np.random.standard_normal((npts,))*n_0_err
        sigma_NT = sigma_NT_C + np.random.standard_normal((npts,))*0.1*sigma_NT_C ## added 10% variation to sigma_{NT} by hand
        rc = rc_ + np.random.standard_normal((npts,))*0.1*rc_ ## added 10% variation to sigma_{NT} by hand  
        n_c = n_0/(1+(rc/r_0)**2)**power
    #    beta = beta_ + np.random.standard_normal((npts,))*0.20*beta_ ## added 30 variation because of simulation results 
    #    beta = np.random.uniform(0.5,0.8,npts)
        B_c = (sigma_NT  *(4* np.pi*mu* np.abs(n_c))**(.5))/beta

        n = n_0/(1+(r/r_0)**2)**power
        B = (B_c *(n/n_c)**k) *10**6
        B_sig = np.std(B)
        B_sigma.append(B_sig)
        B_mean.append(np.mean(B))
        B_1quartile.append(np.percentile(B,25))
        B_3quartile.append(np.percentile(B,75))
        
    return (B_mean,B_sigma,B_1quartile,B_3quartile)

def total_masserror(rc_,r_flatinpc,n_flat,index,Plummermodel_error):
    '''
    Added 30 July 2018 
    This calculated the total mass of the cores which is assumed to be sphere
 
    
    Input are:
    
    r_c = is the transonic radius 
    r_flatinpc  = r_0 or  r_flat in pc
    n_0 = n0 or nflat in cm^-3
    index =  the power of the density function
    
    mass = 4 pi mu (numerical integral)
    ### Important to note that this gives mass in grams ###
    '''
    
    
    npts = 1000
    nsim = 10
    
    totalmassararay = []
    totalmass_mean = []
    totalmass_sigma =[]
    totalmass_1quartile =[]
    totalmass_3quartile =[]
    
    
    eta_err = Plummermodel_error[0]
    r_0_err = Plummermodel_error[1]
    n_0_err = Plummermodel_error[2]
    
    for i in range (1, nsim):
        r_c = rc_ + np.random.standard_normal((npts,))*0.1*rc_ ## added 10% variation to sigma_{NT} by hand 
        power= index + np.random.standard_normal((npts,))*eta_err
        r_0 = r_flatinpc  + np.random.standard_normal((npts,))*r_0_err
        n_0 = n_flat + np.random.standard_normal((npts,))*n_0_err
        
        for j in range(0,npts):
    #         print("random intgration number", j)
            r0=r_0[j-1]
            n0 = n_0[j-1]
            rc = r_c[j-1]
            pow = power[j-1]
#           
            totalmass= mf.totalmass(rc,r0,n0,pow)
            totalmassararay.append(totalmass)
        
        totalmass_mean.append(np.mean(totalmassararay)) 
        totalmass_sigma.append(np.std(totalmassararay))
        totalmass_1quartile.append(np.percentile(totalmassararay,25))
        totalmass_3quartile.append(np.percentile(totalmassararay,75))
        
    #totalmass_interquatilerange = (totalmass_3quartile - totalmass_1quartile)/2)
    
    #     print(masstoflux_mean,masstoflux_1quartile,masstoflux_3quartile)  

#     return(totalmass_mean,totalmass_sigma)
    return(totalmass_1quartile,totalmass_3quartile)



# In[ ]:

## read the magnetic fied error estimate data file:
def readmagneticfielderror(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    B_1quatile_list = []        # units arcsec 
    B_mean_list= []      # units cm^{-3}
    B_3quatile_list = []

    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        B_1quatile_list.append(float(t[0]))
        B_mean_list.append(float(t[1]))
        B_3quatile_list.append(float(t[2])) 

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    B_1quatile= np.asarray(B_1quatile_list)
    B_mean = np.asarray(B_mean_list)
    B_3quatile =np.asarray(B_3quatile_list)
    return (B_1quatile, B_mean,B_3quatile)


# In[ ]:




def masstoflux_sigmaerror(radiusinpcformass,r_flatinpc,rc,n_flat,sigma_NT_C,index,Plummermodel_error,l,k):
    '''
    This function estimate the disribution of m2f at a particular radius 
    Please note we have used beta = 0.5
    Input Parameter : index 
                    : r_flatinpc
                    : n_flat
                    : sigma_NT_c
                    : The fit error of the Plummer Model (added on 23/04/18)
                    : radius at which the field is estiamtes
                    : k the \kappa for the flux freezing model
                    : The transcritical radius rc
    
    '''
    print("radius", radiusinpcformass[l], "simulation is running")
    masstofluxararay = []
    masstoflux_mean = []
    masstoflux_1quartile= []
    masstoflux_3quartile = []
    npts = 100
    nsim = 20
    beta = 0.5 ## the approxiamtion used at rc
    eta_err = Plummermodel_error[0]
    r_0_err = Plummermodel_error[1]
    n_0_err = Plummermodel_error[2]
    for i in range (1, nsim):
#         print("simulation number ", i)
        power= index + np.random.standard_normal((npts,))*eta_err
        r_0 = r_flatinpc  + np.random.standard_normal((npts,))*r_0_err
        n_0 = n_flat + np.random.standard_normal((npts,))*n_0_err
        sigma_NT = sigma_NT_C + np.random.standard_normal((npts,))*0.1*sigma_NT_C

        n_c = n_0/(1+(rc/r_0)**2)**power

        B_c = (sigma_NT  *(4* np.pi*mu* np.abs(n_c))**(.5))/beta

#         n = n_0/(1+(r/r_0)**2)**power

        for j in range(1,npts):
    #         print("random intgration number", j)
            r0=r_0[j-1]
            n0 = n_0[j-1]
            ncc = n_c[j-1]
            Bc = B_c[j-1]
            pow = power[j-1]
#             k=1/2
#             print(l)
            masstoflux= mf.numericalmasstoflux(radiusinpcformass,r_flatinpc,rc,n0,ncc,Bc,pow,k,l)
            masstofluxararay.append(masstoflux)

    #         print(masstoflux)
        masstoflux_mean.append(np.mean(masstofluxararay)) 
        masstoflux_1quartile.append(np.percentile(masstofluxararay,25))
        masstoflux_3quartile.append(np.percentile(masstofluxararay,75))
    #     print(masstoflux_mean,masstoflux_1quartile,masstoflux_3quartile)  

    return (masstoflux_mean,masstoflux_1quartile,masstoflux_3quartile)


# In[ ]:

## read the mass-to-flux error estimate data file:
def readmasstofluxerror(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    mass2flux_1quatile_list = []        # units arcsec 
    mass2flux_mean_list= []      # units cm^{-3}
    mass2flux_3quatile_list = []

    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        mass2flux_1quatile_list.append(float(t[0]))
        mass2flux_mean_list.append(float(t[1]))
        mass2flux_3quatile_list.append(float(t[2])) 

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    mass2flux_1quatile= np.asarray(mass2flux_1quatile_list)
    mass2flux_mean = np.asarray(mass2flux_mean_list)
    mass2flux_3quatile =np.asarray(mass2flux_3quatile_list)
    return (mass2flux_1quatile, mass2flux_mean,mass2flux_3quatile)

def readmasstoflux(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    mass2flux_k1_list = []        # units arcsec
    mass2flux_k2_list = []
    
    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        mass2flux_k1_list.append(float(t[0]))
        mass2flux_k2_list.append(float(t[1]))
        

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    mass2flux_k1= np.asarray(mass2flux_k1_list)
    mass2flux_k2= np.asarray(mass2flux_k2_list)

    return (mass2flux_k1,mass2flux_k2)

def readmagneticfield(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    radiusinpc_list=[]
    density_list = []
    B1_list = []        # units arcsec
    B2_list = []
    B1_ul_list = []
    B2_ul_list = []
    
    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        radiusinpc_list.append(float(t[0]))
        density_list.append(float(t[1]))
        B1_list.append(float(t[2]))
        B2_list.append(float(t[3]))
        B1_ul_list.append(float(t[4]))
        B2_ul_list.append(float(t[5]))
        

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    radiusinpc = np.asarray(radiusinpc_list)
    density = np.asarray(density_list)
    B1 =np.asarray(B1_list)
    B2 = np.asarray(B2_list)
    B1_ul = np.asarray(B1_ul_list)
    B2_ul = np.asarray(B2_ul_list)
    

    return (radiusinpc,density,B1,B2,B1_ul,B2_ul)

def readplummerfitdata(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    n_flat_list=[]
    n_flat_error_list=[]
    r_flatinpc_list = []
    r_flatinpc_error_list = []
    index_list = []
    index_error_list = []
    
    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        n_flat_list.append(float(t[0]))
        n_flat_error_list.append(float(t[1]))
        r_flatinpc_list.append(float(t[2]))
        r_flatinpc_error_list.append(float(t[3]))
        index_list.append(float(t[4]))
        index_error_list.append(float(t[5]))
        

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    n_flat = np.asarray(n_flat_list)
    n_flat_error = np.asarray(n_flat_error_list)
    r_flatinpc=np.asarray(r_flatinpc_list)
    r_flatinpc_error=np.asarray(r_flatinpc_error_list)
    index=np.asarray(index_list)
    index_error=np.asarray(index_error_list)
    

    return ( n_flat,n_flat_error,r_flatinpc,r_flatinpc_error,index,index_error)

def readdeltaBdata(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    radiusforplot_list = []
    deltaB_list=[]
    mean_deltaB_list = []
    deltaB_sigma_list =[]
    deltaBoverB_k1_list=[]
    deltaBoverB_k1_sigma_list=[]
    deltaBoverB_k2_list = []
    deltaBoverB_k2_sigma_list = []
#     r_flatinpc_error_list = []
#     index_list = []
#     index_error_list = []
    
    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        radiusforplot_list.append(float(t[0]))
        deltaB_list.append(float(t[1]))
        mean_deltaB_list.append(float(t[2]))
        deltaB_sigma_list.append(float(t[3]))
        deltaBoverB_k1_list.append(float(t[4]))
        deltaBoverB_k1_sigma_list.append(float(t[5]))
        deltaBoverB_k2_list.append(float(t[6]))
        deltaBoverB_k2_sigma_list.append(float(t[7]))
#         r_flatinpc_error_list.append(float(t[3]))
#         index_list.append(float(t[4]))
#         index_error_list.append(float(t[5]))
        

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    radiusforplot = np.asarray(radiusforplot_list)
    deltaB = np.asarray(deltaB_list)
    mean_deltaB = np.asarray(mean_deltaB_list)
    deltaB_sigma = np.asarray(deltaB_sigma_list)
    deltaBoverB_k1 = np.asarray(deltaBoverB_k1_list)
    deltaBoverB_k1_sigma = np.asarray(deltaBoverB_k1_sigma_list)
    deltaBoverB_k2=np.asarray(deltaBoverB_k2_list)
    deltaBoverB_k2_sigma = np.asarray(deltaBoverB_k2_sigma_list)
#     r_flatinpc_error=np.asarray(r_flatinpc_error_list)
#     index=np.asarray(index_list)
#     index_error=np.asarray(index_error_list)
    

    return (radiusforplot,deltaB,mean_deltaB,deltaB_sigma,deltaBoverB_k1,deltaBoverB_k1_sigma,deltaBoverB_k2,deltaBoverB_k2_sigma)

def readenergydata(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    radiusinpcforenergy_list = []
    kineticenergy_list=[]
    kineticnonthermal_list=[]
    potentialenergy_list = []
    
    magneticenergy_list = []
    
    
    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        radiusinpcforenergy_list.append(float(t[0]))
        kineticenergy_list.append(float(t[1]))
        kineticnonthermal_list.append(float(t[2]))
        potentialenergy_list.append(float(t[3]))
        magneticenergy_list.append(float(t[4]))
       

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    radiusinpcforenergy=np.asarray(radiusinpcforenergy_list)
    kineticenergy = np.asarray(kineticenergy_list)
    kineticnonthermal = np.asarray(kineticnonthermal_list)
    potentialenergy=np.asarray(potentialenergy_list)
    magneticenergy=np.asarray(magneticenergy_list)
    
    

    return (radiusinpcforenergy, kineticenergy,kineticnonthermal,potentialenergy,magneticenergy)


def isopedic_data(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    radiusinpcforplot_list = []
    B_list=[]
    B_sigma_montecarlo_list=[]
    
    
    
    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        radiusinpcforplot_list.append(float(t[0]))
        B_list.append(float(t[1]))
        B_sigma_montecarlo_list.append(float(t[2]))
        
       

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    radiusinpcforplot=np.asarray(radiusinpcforplot_list)
    B = np.asarray(B_list)
    B_sigma_montecarlo = np.asarray(B_sigma_montecarlo_list)
    
    
    

    return (radiusinpcforplot, B,B_sigma_montecarlo)

def readhourgl_mf_data(fileName):
    fp = open (fileName,'r')
    ## Declaring an empty list to store the data from the .dat file
    radiusinpcforplot_list = []
    mf_list=[]
    mf_list_ul=[]
    
    
    
    
    ## Reading the data from a .dat file line by line
    for line in fp:
        t = line.strip().split()  
        radiusinpcforplot_list.append(float(t[0]))
        mf_list.append(float(t[1]))
        mf_list_ul.append(float(t[2]))
        

    fp.close()  

    ##to convert rho python list to numpy array list for easy manipulation
    radiusinpcforplot=np.asarray(radiusinpcforplot_list)
    mf = np.asarray(mf_list)
    mf_list_ul = np.asarray(mf_list_ul)
    
    

    return (radiusinpcforplot,mf,mf_list_ul)