# %%
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad,trapz, cumtrapz, odeint, solve_ivp
from scipy import interpolate
import ThermalFrontFormulation as TF
from unpackConfigurations import unpackConfiguration,returnzl,returnll
from matplotlib.collections import LineCollection
import multiprocessing as mp
from collections import defaultdict
from timeit import default_timer as timer

#Function to integrate, that returns dq/ds and dt/ds using Lengyel formulation and field line conduction
def LengFunc(y,s,kappa0,nu,Tu,cz,qpllu0,alpha,radios,S,B,Xpoint,Lfunc):

    qoverB,T = y
    #set density using constant pressure assumption
    ne = nu*Tu/T
    
    
    fieldValue = 0
    if s > S[-1]:
        fieldValue = B(S[-1])
    elif s< S[0]:
        fieldValue = B(S[0])
    else:
        fieldValue = B(s)
    #add a constant radial source of heat above the X point
    if radios["upstreamGrid"]:
        if s >S[Xpoint]:
            dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) - qpllu0/np.abs(S[-1]-S[Xpoint])
        else:
            dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    else:
        dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    # working on neutral/ionisation model
    dqoverBds = dqoverBds/fieldValue
    dtds = 0
    if radios["fluxlim"]:
        dtds = qoverB*fieldValue/(kappa0*T**(5/2)-qoverB*fieldValue*kappa0*T**(1/2)/(alpha*ne*np.sqrt(9E-31)))
    else:
        dtds = qoverB*fieldValue/(kappa0*T**(5/2))
    #return gradient of q and T
    return [dqoverBds,dtds]


def returnImpurityFracLengMKold(constants,radios,S,indexRange,dispBassum = False,dispqassum = False,dispUassum = False, verbose = False, tol = 1e-5, acceleration = 2):
    """ function that returns the impurity fraction required for a given temperature at the target. Can request a low temperature at a given position to mimick a detachment front at that position."""
    t0 = timer()
    C = []
    radfraction = []
    Tprofiles = []
    Sprofiles = []
    Qprofiles = []
    splot = []
    logs = []
    error1 = 1
    error0 = 1

    #lay out constants
    gamma_sheath = constants["gamma_sheath"]
    qpllu0 = constants["qpllu0"]
    nu = constants["nu"]
    kappa0 = constants["kappa0"]
    mi = constants["mi"]
    echarge = constants["echarge"]
    Tt = constants["Tt"]
    Xpoint = constants["XpointIndex"]
    B = constants["B"]
    Lfunc = constants["Lfunc"]
    alpha = constants["alpha"]

    #initialise arrays for storing cooling curve data
    Tcool = np.linspace(0.3,500,1000)#should be this for Ar? Ryoko 20201209 --> almost no effect
    Lalpha = []
    for dT in Tcool:
        Lalpha.append(Lfunc(dT))
    Lalpha = np.array(Lalpha)
    Tcool = np.append(0,Tcool)
    Lalpha = np.append(0,Lalpha)
    Lz = [Tcool,Lalpha]

    print("Solving...", end = "")
    
    # Define iterator function. This just solves the Lengyel function and unpacks the results.
    def iterate(cz, Tu):
        result = odeint(LengFunc,y0=[qpllt/B(s[0]),Tt],t=s,args=(kappa0,nu,Tu,cz,qpllu0,alpha,radios,S,B,Xpoint,Lfunc))
        out = dict()
        # Result returns integrals of [dqoverBds, dtds]
        out["q"] = result[:,0]*B(s)
        out["T"] = result[:,1]
        out["qpllu1"] = out["q"][-1]
        out["Tu"] = out["T"][-1]

        if radios["upstreamGrid"]:
            out["error1"] = (out["qpllu1"]-0)/qpllu0
        else:
            out["error1"] = (out["qpllu1"]-qpllu0)/qpllu0

        return out

    for point in indexRange:

        ##### INITIAL GUESSES
        
        # Current set of parallel position coordinates
        s = S[point:]
        splot.append(S[point])

        # Inital guess for the value of qpll integrated across connection length
        qavLguess = 0
        if radios["upstreamGrid"]:
            if s[0] < S[Xpoint]:
                qavLguess = ((qpllu0)*(S[Xpoint]-s[0]) + (qpllu0/2)*(s[-1]-S[Xpoint]))/(s[-1]-S[0])
            else:
                qavLguess = (qpllu0/2)
        else:
            qavLguess = (qpllu0)

        # Inital guess for upstream temperature based on guess of qpll ds integral
        Tu0 = ((7/2)*qavLguess*(s[-1]-s[0])/kappa0)**(2/7)
        Tu = Tu0
        
        # Cooling curve integral
        Lint = cumtrapz(Lz[1]*np.sqrt(Lz[0]),Lz[0],initial = 0)
        integralinterp = interpolate.interp1d(Lz[0],Lint)

        # Initial guess of cz0 assuming qpll0 everywhere and qpll=0 at target
        cz0 = (qpllu0**2 )/(2*kappa0*nu**2*Tu**2*integralinterp(Tu))
        cz = cz0
            
        
        # Initial guess of qpllt, typically 0
        qpllt = gamma_sheath/2*nu*Tu*echarge*np.sqrt(2*Tt*echarge/mi)
        
        ##### INITIALISATION
        log = defaultdict(list)
        error1 = 1
        error0 = 1
        log["error1"].append(error1)
        
        
        while abs(error0) > tol: # Tu convergence loop
            # Initial guesses
            

            # Initialise
            
            out = iterate(cz, Tu)
            if verbose:
                print("cz: {:.2f}, error1: {:.4E}".format(cz, out["error1"]))

            ##### INITIAL SOLUTION BOUNDING
            # We are either doubling or halving cz until the error flips sign
            log["cz"].append(cz)
            log["error1"].append(out["error1"])

            if out["error1"] > 0:
                while out["error1"] > 0:
                    cz = cz / 2
                    out = iterate(cz, Tu)
                    log["cz"].append(cz)
                    log["error1"].append(out["error1"])
                    if verbose:
                        print("cz: {:.2f}, error1: {:.3E}".format(cz, out["error1"]))
            else:
                while out["error1"] < 0:
                    cz = cz * 2
                    out = iterate(cz, Tu)
                    log["cz"].append(cz)
                    log["error1"].append(out["error1"])
                    if verbose:
                        print("cz: {:.2f}, error1: {:.3E}".format(cz, out["error1"]))

            # We have bounded the problem -  the last two iterations
            # are on either side of the solution
            lower_bound = min(log["cz"][-1], log["cz"][-2])
            upper_bound = max(log["cz"][-1], log["cz"][-2])
            
            lower_error = log["error1"][log["cz"].index(lower_bound)+1]
            upper_error = log["error1"][log["cz"].index(upper_bound)+1]

            if acceleration > 0:
    
                if verbose:
                    print("Bounds centering enabled, set to {} iterations".format(acceleration))
                    print("-->Before centering: {:.3f}-{:.3f}".format(lower_bound, upper_bound))

                if abs(upper_error/lower_error) > 10:
                    for k in range(acceleration):
                        upper_bound -= (upper_bound-lower_bound)/2
                elif abs(upper_error/lower_error) < 0.1:
                    for k in range(acceleration):
                        lower_bound += (upper_bound-lower_bound)/2

                if verbose:
                    print("-->After centering: {:.3f}-{:.3f}".format(lower_bound, upper_bound))


            ##### BISECTION SEARCH
            while abs(out["error1"]) > tol:
                # New cz guess is halfway between the upper and lower bound.
                cz = lower_bound + (upper_bound-lower_bound)/2
                out = iterate(cz, Tu)
                log["cz"].append(cz)
                log["error1"].append(out["error1"])

                # Narrow bounds based on the results.
                if out["error1"] < 0:
                    lower_bound = cz
                elif out["error1"] > 0:
                    upper_bound = cz

                if verbose:
                    print(">Bounds: {:.3f}-{:.3f}, cz: {:.3f}, error1: {:.3E}".format(
                        lower_bound, upper_bound, cz, out["error1"]))

            # Calculate the new Tu by mixing half the old and half the new value.
            Tucalc = out["Tu"]
            Tu = 0.5*Tu + 0.5*Tucalc
            error0 = (Tu-Tucalc)/Tu

            if verbose:
                print("-----------error0: {:.3E}, Tu: {:.2f}\n".format(error0, Tu))
                
            
            log["Tu"].append(Tu)
      
            log["error0"].append(error0)
            
            Q = []
            for Tf in out["T"]:
                Q.append(Lfunc(Tf))
                
        
        C.append(np.sqrt(cz))
        Tprofiles.append(out["T"])
        Sprofiles.append(s)
        Qprofiles.append(out["q"])
        
        print("{}...".format(point), end="")    
        logs.append(log)
    
    t1 = timer()
    
    print("Complete in {:.1f} seconds".format(t1-t0))
        
    return splot, C, Sprofiles,Tprofiles,Qprofiles,logs
