
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad


# In[1]:

def masstheorycalculation(u,uc,n_0,r_flatinpc,i): 
    
    '''Theoretical Calculation of Mass M(u) 
       Only valid for index 1 in the density formula 
       u = r/r_flat 
       uc = r_c /r_flat, where r_c is the radius of the core
       M (u) = 4 pi m n_o r_flat**3 (m2)  # units g  
       num1 = (uc - arctan(uc)) 
       num2 = sqrt(u^2 +1)[sqrt((uc^2-u^2)/(u^2+1)) - arctan(sqrt((uc^2-u^2)/(u^2+1)))] 
       m2 = num1 - num2
       
    '''
#   Constants
    amu = 1.6605402e-24 # g
    mu = 2.33*amu
    #  m1 gives the total mass of the sphere and i takes only the max value
    m1  = (u[i] - np.arctan(u[i]))   # this is just for testing m2
    
    # The mass of the sphere as a fucntion of u   
    m2  = ((uc-np.arctan(uc)) - (np.sqrt(u[i]**2+1)*(np.sqrt((uc**2-u[i]**2)/(u[i]**2+1))-
                                                          np.arctan(np.sqrt((uc**2-u[i]**2)/(u[i]**2+1))))))
    
    mass1 =  4 * np.pi * mu * n_0 * (r_flatinpc*3.0857e+18)**3 * m1
    mass = 4 * np.pi * mu * n_0 * (r_flatinpc*3.0857e+18)**3 * m2
#     return(mass,mass1)
    return(mass)

def constantformass2flux(r_flatinpc,sigma_NT_C,n_0,n_c,beta):
    '''
    constant = ((4 pi m n_o r_flat**3) * 2 pi sqrt (G) )/  (2 pi B_c r_flat**2)
    
    '''
     
    amu = 1.6605402e-24 # g
    mu = 2.33*amu
    G = 6.67259e-8 # cm3 g-1 s-2
    
    constant = beta*(r_flatinpc*3.0857e+18/sigma_NT_C) *(np.sqrt((4* np.pi*mu* G * (n_0**2) )/ n_c))
    return (constant)





def masstoflux(u,n_0,n_c,index,k,i,constant,uc): 
    
    
    '''
       Theoretical Calculation of masstoflux normalised i.e. (M/\phi)/(M/\phi)_0 
       where (M/\phi)_0  = 1/2\pi G**.5
       Only valid for index 1 in the density formula 
       u = r/r_flat 
       uc = r_c /r_flat, where r_c is the radius of the core
       M (u) = 4 pi m n_o r_flat**3 (m2)  # units g  
       num1 = (uc - arctan(uc)) 
       num2 = sqrt(u^2 +1)[sqrt((uc^2-u^2)/(u^2+1)) - arctan(sqrt((uc^2-u^2)/(u^2+1)))] 
       m2 = num1 - num2
       
       phi = 2 \pi B_c r_flat**2 \integral (u (n(u)/n_c)*k du) 
       
       n(u) = (n_0/(1+u**2))**index # where the index is one
       
    '''
    def flux(u,n_0,n_c,index,k):
        n = (n_0/(1+u**2))**index
        flux = u* ((n/n_c)**(k))
#         flux = u* ((n/n_c)**(0.5-k))
        return(flux)
   
    a = n_0
    b = n_c
    c = index
    d = k
    
    # calculating the flux
    I = quad(flux, min(u),u[i], args=(a,b,c,d))
    #     print(I)
    
    
    # masstoflux for the total sphere. This is just to test the final value
    mass2flux1  = constant *(u[i] - np.arctan(u[i]))/(I[0])
    
    # defining mass at each radius
    m2  = ((uc-np.arctan(uc)) - (np.sqrt(u[i]**2+1)*(np.sqrt((uc**2-u[i]**2)/(u[i]**2+1))-
                                                          np.arctan(np.sqrt((uc**2-u[i]**2)/(u[i]**2+1))))))
    
    
    mass2flux2  = (constant * m2)/(I[0])
    return(mass2flux2)


# In[3]:

def numericalmassforanyindex(radiusinpc,r_c,r_flatinpc,n_0,index,i):
    '''
    This function numerically evaluates the mass of the sphere.
    Input are:
    radiusinpc[i] = radius as which the mass is required =
    r_c = is the transonic radius "Note radius radiusinpc[i] can not be more than r_c "
    r_flatinpc  = r_0 or  r_flat in pc
    n_0 = n0 or nflat in cm^-3
    i indicate the value in the radiusinpc array
    index =  the power of the density function
    
    mass = 2 pi mu (numerical integral)
    ### Important to note that this gives mass in grams ###
    '''
    
    import numpy as np
    from scipy import integrate
    def f(r, x):
        amu = 1.6605402e-24 # g
        mu = 2.33*amu
        r0 = r_flatinpc* 3.0857e+18
        n0 =  n_0
        constant = 2 * np.pi * mu
        density = (n0/(1+(r/r0)**2)**index)
        return (constant * 2* x*density *r /(np.sqrt(r**2 - x**2)))
    
    r1 = r_c *3.0857e+18    # this is the upper limit of sigma integral over r = r_c the transonic radius
    r2 = radiusinpc *3.0857e+18  # this is the radius at which we want the mass
 
    def bounds_x():
        return [0, r2[i]]

    def bounds_r(x):
        return [x, r1]

    result = integrate.nquad(f, [bounds_r, bounds_x])
    return (result[0]) #  mass in grams 


# In[ ]:

def numericalflux(radiusinpc,r_flatinpc,n_0,n_c,B_c,index,k,i):
    
    
    '''
    This numerically evaluates the flux as function of radius
    
    '''
    
    def f(r):
        
        constant = 2 *np.pi *B_c
        
        r0 = r_flatinpc* 3.0857e+18
        nc = n_c
        n = (n_0/(1+(r/r0)**2))**index
        flux = constant* r* ((n/nc)**(k))
        return(flux)
    
    rlimit = radiusinpc * 3.0857e+18
    print(i)
     # where B_c is in gauss
    def bounds_r():
        return [0, rlimit[i]]
    
    result = integrate.nquad(f, [bounds_r])  
#     return ("the flux is %e" %result[0],'gauss cm^{2}') # units gauss cm**2 Flux
    return(result[0]) # units gauss cm**2 Flux


# In[ ]:

def numericalmasstoflux(radiusinpc,r_flatinpc,r_c,n_0,n_c,B_c,index,k,i):
    
    G = 6.67259e-8 # cm3 g-1 s-2
    amu = 1.6605402e-24 # g
    mu = 2.33*amu
    
    import numpy as np
    from scipy import integrate
    def m(r, x):
#         amu = 1.6605402e-24 # g
#         mu = 2.33*amu
        r0 = r_flatinpc* 3.0857e+18
        n0 =  n_0
        constant = 2 * np.pi * mu
        density = (n0/(1+(r/r0)**2)**index)
        return (constant * 2* x*density *r /(np.sqrt(r**2 - x**2)))
    
    r1 = r_c *3.0857e+18    # this is the upper limit of sigma integral over r = r_c the transonic radius
    r2 = radiusinpc *3.0857e+18  # this is the radius at which we want the mass
 
    def bounds_x():
        return [0, r2[i]]

    def bounds_r(x):
        return [x, r1]

    resultmass = integrate.nquad(m, [bounds_r, bounds_x])
#     return (resultmass[0]) #  mass in grams 
    

    def f(r):
        
        constant = 2 *np.pi *B_c
        
        r0 = r_flatinpc* 3.0857e+18
        nc = n_c
        n0 =  n_0
        density = (n0/(1+(r/r0)**2)**index)
        flux = constant* r* ((density/nc)**(k))
        return(flux)
    
    rlimit = radiusinpc * 3.0857e+18
    
     # where B_c is in gauss
    def bounds_r():
        return [0, rlimit[i]]
    
    resultflux = integrate.nquad(f, [bounds_r])  
#     return(resultflux[0]) # units gauss cm**2 Flux

    m2f0 = 1/(2* np.pi *np.sqrt(G))
#     print(m2f0)
    m2f = resultmass[0]/resultflux[0]
    m2fnormalised = m2f / m2f0 
    
    return(m2fnormalised)


# In[ ]:

# function for reading the transonic radius and sigma after interpolatio
def readfilefortransonicdata(filename):
    fp = open (filename,'r')
    readline = []
    for line in fp:
        readline.append(float(line))
    fp.close() 

    return(readline)



def totalmass(r_c,r_flatinpc,n_0,index):
    '''
    Added 30 July 2018 
    This calculated the total mass of the cores which is assumed to be sphere
 
    
    Input are:
    
    r_c = is the transonic radius "Note radius radiusinpc[i] can not be more than r_c "
    r_flatinpc  = r_0 or  r_flat in pc
    n_0 = n0 or nflat in cm^-3
    
    index =  the power of the density function
    
    mass = 4 pi mu (numerical integral)
    ### Important to note that this gives mass in grams ###
    '''
    
    import numpy as np
    from scipy import integrate
    def f(r):
        amu = 1.6605402e-24 # g
        mu = 2.33*amu
        r0 = r_flatinpc* 3.0857e+18
        n0 =  n_0
        constant = 4 * np.pi * mu
        density = (n0/(1+(r/r0)**2)**index)
#         return (constant * 2* x*density *r /(np.sqrt(r**2 - x**2)))
        return (constant* density * r**2)
    
    r1 = r_c *3.0857e+18    # this is the upper limit of sigma integral over r = r_c the transonic radius
#     r2 = radiusinpc *3.0857e+18  # this is the radius at which we want the mass
 
    def bounds_r():
        return [0, r1]

#     def bounds_r(x):
#         return [x, r1]

    result = integrate.nquad(f, [bounds_r])
    return (result[0]) #  mass in grams 


## to calculate the mean magnetic field: 

def mean_magnetic_field(r_c,B_cc,r_flatinpc,n_0,nc,index,k):
    '''
    Added 12 March 2019 for the JCMT proposal 
    This calculated the mean B of the cores which is assumed to be sphere
 
    
    Input are:
    
    r_c = is the transonic radius 
    r_flatinpc  = r_0 or  r_flat in pc
    n_0 = n0 or nflat in cm^-3
    k = the power-law index
    index =  the power of the density function
    
    mean B = (4 pi) / (4/3 pi R^3) integral (B(r) r^2 dr) 
    
    '''
#     print(rc,B_cc,r_flatinpc,n_flat,nc,index)
    import numpy as np
    from scipy import integrate
    def f(r):
        r0 = r_flatinpc 
        n0 =  n_0 # n_flat
        constant = (3 / ((r_c)**3))
        density = (n0/(1+(r/r0)**2)**index)
        B_total = (B_cc/(nc**(k)))* density**(k)
#         return (constant * 2* x*density *r /(np.sqrt(r**2 - x**2)))
        return (constant* B_total * r**2)
    
    r1 = r_c    # this is the upper limit of sigma integral over r = r_c the transonic radius
#     r2 = radiusinpc *3.0857e+18  # this is the radius at which we want the mass
 
    def bounds_r():
        return [0, r1]

    result = integrate.nquad(f, [bounds_r])
    return (result[0]) #  mass in micro gauss 
