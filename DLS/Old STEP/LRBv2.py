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
import pandas as pd
import sys

"""
This version is for implementing density and power as a detachment front driver
"""

#Function to integrate, that returns dq/ds and dt/ds using Lengyel formulation and field line conduction
def LengFunc(y,s,kappa0,nu,Tu,cz,qpllu0,alpha,radios,S,B,Xpoint,Lfunc,qradial, qradial_fix):

    qoverB,T = y
    #set density using constant pressure assumption
    ne = nu*Tu/T
    
    fieldValue = 0
    if s > S[-1]:
        fieldValue = B(S[-1])
    elif s < S[0]:
        fieldValue = B(S[0])
    else:
        fieldValue = B(s)
        
    # add a constant radial source of heat above the X point, which is qradial = qpll at Xpoint/np.abs(S[-1]-S[Xpoint]
    # i.e. radial heat entering SOL evenly spread between midplane and xpoint needs to be sufficient to get the 
    # correct qpll at the xpoint.
    
    
    
    if radios["upstreamGrid"]:
        if s >S[Xpoint]:
            # The second term here converts the x point qpar to a radial heat source acting between midplane and the xpoint
            if qradial_fix == True:
                try:
                    dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) - qradial * fieldValue / B(S[Xpoint]) # account for flux expansion to Xpoint
                except:
                    print("Failed. s: {:.2f}".format(s))
            else:
                dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) - qradial  
        else:
            dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    else:
        dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    
    # working on neutral/ionisation model
    dqoverBds = dqoverBds/fieldValue
    
    # Flux limiter
    dtds = 0
    if radios["fluxlim"]:
        dtds = qoverB*fieldValue/(kappa0*T**(5/2)-qoverB*fieldValue*kappa0*T**(1/2)/(alpha*ne*np.sqrt(9E-31)))
    else:
        dtds = qoverB*fieldValue/(kappa0*T**(5/2))
    #return gradient of q and T
    return [dqoverBds,dtds]


def LRBv2(constants,radios,S,indexRange, 
                             control_variable = "impurity_frac",
                             qradial_fix = True,
                             verbosity = 0, Ctol = 1e-3, Ttol = 1e-2, 
                             acceleration = 0, URF = 0.9,
                             timeout = 20):
    """ function that returns the impurity fraction required for a given temperature at the target. Can request a low temperature at a given position to mimick a detachment front at that position.
    constants: dict of options
    radios: dict of options
    S: S parallel array
    indexRange: array of S indices of the parallel front locations to solve for
    control_variable: either impurity_frac, density or power
    qradial_fix: accounts for flux expansion in the heating region (midplane to Xpoint)
    Ctol: error tolerance target for the inner loop (i.e. density/impurity/heat flux)
    Ttol: error tolerance target for the outer loop (i.e. rerrunning until Tu convergence)
    acceleration: makes bisection method faster for the inner loop by pre-centering bounds
    URF: under-relaxation factor for temperature. If URF is 0.2, Tu_new = Tu_old*0.8 + Tu_calculated*0.2
    Timeout: controls timeout for all three loops within the code. Each has different message on timeout
    
    """
    t0 = timer()
    C = []
    radfraction = []
    splot = []
    error1 = 1
    error0 = 1
    output = defaultdict(list)

    #lay out constants
    gamma_sheath = constants["gamma_sheath"]
    qpllu0 = constants["qpllu0"]
    nu0 = constants["nu0"]
    cz0 = constants["cz0"]
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
    
    # Calculation of radial heat transfer needed to achieve correct qpllu0 at Xpoint
    qradial = qpllu0/np.abs(S[-1]-S[Xpoint])

    print("Solving...", end = "")
    
    # Define iterator function. This just solves the Lengyel function and unpacks the results.
    def iterate(cvar, Tu):
        if control_variable == "impurity_frac":
            cz = cvar
            nu = nu0
            qradial = qpllu0/np.abs(S[-1]-S[Xpoint]) # Change Q radial so it takes B field into account
            
        elif control_variable == "density":
            cz = cz0
            nu = cvar
            qradial = qpllu0/np.abs(S[-1]-S[Xpoint])   
            
        elif control_variable == "power":
            cz = cz0
            nu = nu0
            qradial = 1/cvar # This is needed so that too high a cvar gives positive error
        
        if verbosity>2:
            print(f"\nIN--> qpllu0: {qpllu0:.3E} | nu: {nu:.3E} | Tu: {Tu:.1f} | cz: {cz:.3E}")
        
        result = odeint(LengFunc,y0=[qpllt/B(s[0]),Tt],t=s,args=(kappa0,nu,Tu,cz,qpllu0,alpha,radios,S,B,Xpoint,Lfunc,qradial,qradial_fix))
        out = dict()
        # Result returns integrals of [dqoverBds, dtds]
        out["q"] = result[:,0]*B(s)
        out["T"] = result[:,1]
        # out["T"] = np.array(pd.DataFrame(out["T"]).interpolate()) # fill NaNs created by odeint
        # out["q"] = np.array(pd.DataFrame(out["q"]).interpolate()) # fill NaNs created by odeint
        out["qpllu1"] = out["q"][-1]
        out["Tu"] = out["T"][-1]
        Tucalc = out["Tu"]
        
        qpllu1 = out["qpllu1"]
        
        # print(out["T"])
        # print("{:.3E}".format(out["qpllu1"]))
        
        # If upstream grid, qpllu1 is at the midplane and is solved until it's 0. It then gets radial transport
        # so that the xpoint Q is qpllu0. If uypstramGrid=False, qpllu1 is solved to match qpllu0 at the Xpoint.
        if radios["upstreamGrid"]:
            out["error1"] = (out["qpllu1"]-0)/qpllu0 
        else:
            out["error1"] = (out["qpllu1"]-qpllu0)/qpllu0
        
        if verbosity > 2:
            print(f"OUT qpllu1: {qpllu1:.3E} | Tucalc: {Tucalc:.1f}")
        
        return out

    for point in indexRange:
        print("{}...".format(point), end="")    
        
        if verbosity > 0:
            print("\n---SOLVING FOR INDEX {}".format(point))
            
        """------INITIAL GUESSES"""
        
        # Current set of parallel position coordinates
        s = S[point:]
        output["Splot"].append(S[point])

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

        # Guesses/initialisations for control variables
        if control_variable == "impurity_frac":
            # Initial guess of cz0 assuming qpll0 everywhere and qpll=0 at target
            # Initialise control variable as the impurity fraction guess
            cz0_guess = (qpllu0**2 )/(2*kappa0*nu0**2*Tu**2*integralinterp(Tu))
            cvar = cz0_guess
            
        elif control_variable == "density":
            # Initialise control variable as the starting density
            # Impurity fraction is set to constant as cz0
            cvar = nu0
        
        elif control_variable == "power":
            # If control variable = Power, keep nu0 and cz0 constant
            # Initialise control variable 1/qradial. This is needed so that
            # too high a cvar results in a positive error!
            cvar = 1/qradial #qpllu0
            
        # Initial guess of qpllt, typically 0. This is a guess for q at the virtual
        # target (cold end of front). It is a very rough approximation since we are assuming
        # a sheath transmission coefficient (gamma) of a real target.
        qpllt = gamma_sheath/2*nu0*Tu*echarge*np.sqrt(2*Tt*echarge/mi)
        
        
        """------INITIALISATION"""
        
        log = defaultdict(list)
        error1 = 1 # Inner loop error (error in qpllu based on provided cz/ne)
        error0 = 1 # Outer loop residual in upstream temperature
        log["error1"].append(error1)
        
        # Tu convergence loop
        for k0 in range(timeout):
            
            # Initialise
            out = iterate(cvar, Tu)
            reverse_search = False
            if verbosity > 1:
                print("\ncvar: {:.3E}, error1: {:.3E}".format(cvar, out["error1"]))

            """------INITIAL SOLUTION BOUNDING"""

            # We are either doubling or halving cvar until the error flips sign
            log["cvar"].append(cvar)
            log["error1"].append(out["error1"])
            log["qpllu1"].append(out["qpllu1"])
            
            for k1 in range(timeout*2):
                
                if out["error1"] > 0:
                    if reverse_search:
                        cvar = cvar * 2
                    else:
                        cvar = cvar / 2
                        
                elif out["error1"] < 0:
                    if reverse_search:
                        cvar = cvar / 2
                    else:
                        cvar = cvar * 2

                out = iterate(cvar, Tu)

                log["cvar"].append(cvar)
                log["error1"].append(out["error1"])
                log["qpllu1"].append(out["qpllu1"])

                if verbosity > 1:
                    print("cvar: {:.3E}, error1: {:.3E}".format(cvar, out["error1"]))
    
                if verbosity > 2:
                    print("Last error: {:.3E}, New error: {:.3E}".format(log["error1"][k1+1], log["error1"][k1+2]))

                if np.sign(log["error1"][k1+1]) != np.sign(log["error1"][k1+2]): # It's initialised with a 1 already, hence k1+1 and k1+2
                    break
                    
                if k1 == timeout - 1:
                    print("******INITIAL BOUNDING TIMEOUT! Saturation!. Set verbosity = 3!*******")
                    print(f"qpllu0: {qpllu0} | qpllu1: {qpllu1}")
                    # cvar = log["cvar"][0]
                    # reverse_search = True
                    # sys.exit()

            if cvar < 1e-6 and control_variable == "impurity_fraction":
                print("*****REQUIRED IMPURITY FRACTION IS NEAR ZERO*******")
                #sys.exit()
                
            # We have bounded the problem -  the last two iterations
            # are on either side of the solution
            lower_bound = min(log["cvar"][-1], log["cvar"][-2])
            upper_bound = max(log["cvar"][-1], log["cvar"][-2])

            lower_error = log["error1"][log["cvar"].index(lower_bound)+1]
            upper_error = log["error1"][log["cvar"].index(upper_bound)+1]

            # The solution often ends up very near one bound, meaning the other is far away
            # We can pre-narrow this band by x halvings where x = acceleration
            # This can make the bound miss the solution and timeout.
            if acceleration > 0: 

                if verbosity > 1:
                    print("Bounds centering enabled, set to {} iterations".format(acceleration))
                    print("-->Before centering: {:.3E}-{:.3E}".format(lower_bound, upper_bound))

                if abs(upper_error/lower_error) > 10:
                    for k in range(acceleration):
                        upper_bound -= (upper_bound-lower_bound)/2
                elif abs(upper_error/lower_error) < 0.1:
                    for k in range(acceleration):
                        lower_bound += (upper_bound-lower_bound)/2

                if verbosity > 1:
                    print("-->After centering: {:.3E}-{:.3E}".format(lower_bound, upper_bound))


            """------INNER LOOP"""

            for k2 in range(timeout):

                # New cvar guess is halfway between the upper and lower bound.
                cvar = lower_bound + (upper_bound-lower_bound)/2
                out = iterate(cvar, Tu)
                log["cvar"].append(cvar)
                log["error1"].append(out["error1"])
                log["qpllu1"].append(out["qpllu1"])

                # Narrow bounds based on the results.
                if out["error1"] < 0:
                    lower_bound = cvar
                elif out["error1"] > 0:
                    upper_bound = cvar

                if verbosity > 1:
                    print(">Bounds: {:.3E}-{:.3E}, cvar: {:.3E}, error1: {:.3E}".format(
                        lower_bound, upper_bound, cvar, out["error1"]))

                if abs(out["error1"]) < Ctol:
                    break

                if k2 == timeout - 1:
                    print("******INNER LOOP TIMEOUT! Reduce acceleration factor or loosen Ctol. Set verbosity = 2!*******")
                    #sys.exit()
                    
            # Calculate the new Tu by mixing half the old and half the new value.
            Tucalc = out["Tu"]
            Tu = (1-URF)*Tu + URF*Tucalc
            error0 = (Tu-Tucalc)/Tu
            
            if verbosity > 0 :
                print("-----------error0: {:.3E}, Tu: {:.2f}, Tucalc: {:.2f}".format(error0, Tu, Tucalc))
                
            
            log["Tu"].append(Tu)
            log["error0"].append(error0)
            
            # Not sure if this Q serves any function
            Q = []
            for Tf in out["T"]:
                try:
                    Q.append(Lfunc(Tf))
                except:
                    print(f"FAILED TO QUERY COOLING CURVE for a temperature of {Tf:.3E}!")
                    break
                
            if abs(error0) < Ttol:
                break

            if k0 == timeout - 1:
                print("******OUTER TIMEOUT! Loosen Ttol or reduce under-relaxation factor. Set verbosity = 2!*******")
                #sys.exit()

                
        if control_variable == "power":
            output["cvar"].append(1/cvar) # so that output is in Wm-2
        else:
            output["cvar"].append(cvar)
            
        output["Tprofiles"].append(out["T"])
        output["Sprofiles"].append(s)
        output["Qprofiles"].append(out["q"])
        
        Qrad = []
        for Tf in out["T"]:
            if control_variable == "impurity_frac":
                Qrad.append(((nu0**2*Tu**2)/Tf**2)*cvar*Lfunc(Tf))
            elif control_variable == "density":
                Qrad.append(((cvar**2*Tu**2)/Tf**2)*cz0*Lfunc(Tf))
            elif control_variable == "power":
                Qrad.append(((nu0**2*Tu**2)/Tf**2)*cz0*Lfunc(Tf))
            
        output["Rprofiles"].append(Qrad)
        output["logs"].append(log)
        
    # Relative C variable
    cvar_list = np.array(output["cvar"])
    crel_list = cvar_list / cvar_list[0]
    splot = output["Splot"]
    
    # Trim negative gradient
    crel_list_trim = crel_list.copy()
    cvar_list_trim = cvar_list.copy()

    # Find values on either side of C = 1 and interpolate onto 1 
    for i, val in enumerate(crel_list):
        if crel_list[i] < 1 and i < len(crel_list)-1 and crel_list[i+1] >= 1.0:
            interp = interpolate.interp1d([crel_list[i], crel_list[i+1]], [splot[i], splot[i+1]])
            spar_onset = float(interp(1))
            break
        if i == len(crel_list)-1:
            spar_onset = 0

    if len(crel_list)>1:
        grad = np.gradient(crel_list)
        for i, val in enumerate(grad):
            if i > 0 and grad[i] > 0 and grad[i-1] < 0:
                crel_list_trim[:i] = np.nan
                cvar_list_trim[:i] = np.nan
            
    output["splot"] = splot
    output["indexRange"] = indexRange    
    output["cvar"] = cvar_list
    output["crel"] = crel_list
    output["cvar_trim"] = cvar_list_trim
    output["crel_trim"] = crel_list_trim
    output["threshold"] = cvar_list[0]
    output["window"] = cvar_list[-1] - cvar_list[0]
    output["window_ratio"] = cvar_list[-1] / cvar_list[0]
    output["spar_onset"] = spar_onset
    output["constants"] = constants
    output["radios"] = radios
    
    # Convert back to regular dict
    output = dict(output)
    t1 = timer()
    
    print("Complete in {:.1f} seconds".format(t1-t0))
        
    return output

