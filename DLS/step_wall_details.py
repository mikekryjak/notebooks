from geqdsk import *
from freegs import critical as critical
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline 
from scipy.optimize import minimize_scalar
import netCDF4
import shapely.geometry as shp
from SPR_wall import SPR_wall

# Plasma constants
mu0 = 1.2566e-6

def interp2d(R,Z,field):
    return RectBivariateSpline(R,Z,np.transpose(field))

class wall_details:
    def __init__(self,gfile=None,wfile=None,debug=False,monoblock=None,inner=False,Psep=150E6,IP=20E6,aminor=2.0,
                 fig=None, ax=None, custom_plot = False):
        self.debug=debug
        self.Psep=Psep
        self.IP=IP
        self.am=aminor
        self.wfile=wfile
        self.gfile=gfile
        self.read_wall()
        self.read_equil()
        self.calc_bfield()
        self.fig = fig
        self.ax = ax
        self.custom_plot = custom_plot
        if inner:
            self.xpt_dist=-5e-2
        else:
            self.xpt_dist=5e-3        
        self.get_field_angle()
        # if monoblock is None:
        #     # g = tile gap
        #     # l = tile height
        #     # h = misalignment height
        #     # p = perimeter of strike zone
        #     self.monoblock = {"g":1e-3,"l":2e-3,"h":0.3e-3,"p":0.14}
        # else:
        #     self.monoblock = monoblock
        # self.divertor_tiles()
        # self.details()
    def details(self):
        print('Strike point R/Z= ',self.rdiv,self.zdiv)
        print('Poloidal angle  = ',self.pol_angle)
        if np.sign(self.pol_angle) > 0:
            print('Horizontal target geometry')
        else:
            print('Vertical target geometry') 
        print('Incidence angle = ',self.incd_angle)
        print('Field line angle= ',self.theta)
        print('Wetted fraction (toroidal)= ',self.wf)
        print('Unmitigated load on divertor (no tilt) [GW/m^2]= ',self.qsur_notilt/1e9)
        print('Unmitigated load on divertor [GW/m^2]= ',self.qsur/1e9)
        print('Multiplier on SOLPS heat loads =',self.qsur/self.qsur_notilt)
        print('Minor radius / kappa / Bp / R0 = ',self.am,self.kp,self.Bp,self.eq["rcen"])
        print('lambda_q [mm] =',self.lq/1e-3)
        print('Power to divertor [MW] =',self.Psep*self.fdiv/1e6)
        
    def read_wall(self):
        
        sprw = SPR_wall()
        sprw.SPR45new()
        rwall, zwall = sprw.R, sprw.Z
        
        # rwall,zwall = np.loadtxt(self.wfile,unpack=True)
        
        self.wall_data = np.zeros((len(rwall),2))
        self.wall_data[:,0] = rwall
        self.wall_data[:,1] = zwall
        self.shapely_wall = shp.Polygon(self.wall_data)
    
    def read_equil(self):
        # Read the equilibrium
        equil = Geqdsk(self.gfile)

        # Parse the data from the eqdsk file
        psi_bnd = equil['sibry']
        psi_axis = equil['simag']
        psi = equil['psirz']
        cpasma = equil['current']
        btcen = equil['bcentr']
        rcen  = equil['rcentr']
        rbtcen = btcen*rcen
        r_axis = equil['rmaxis']
        z_axis = equil['zmaxis']
        nr = equil['nw']
        nz = equil['nh']

        # Generate the radial and vertical coordinate axes
        R = np.arange(equil['nw'])*equil['rdim']/float(equil['nw'] - 1) + equil['rleft'] 
        Z = np.arange(equil['nh'])*equil['zdim']/float(equil['nh'] - 1) + equil['zmid'] - 0.5*equil['zdim']
        psi_func = interp2d(R,Z,psi)
        
        rr, zz = np.meshgrid(R,Z)
        if self.shapely_wall is not None:
            psi_mask = rr*0.0

            for i in np.arange(np.shape(rr)[0]):
                for j in np.arange(np.shape(rr)[1]):
                    psi_mask[i,j] = self.shapely_wall.contains(shp.Point(rr[i,j],zz[i,j]))
                    
            psi_mask[psi_mask == 0] = np.nan
        else:
            psi_mask = 1.0
        # Calculate the normalised flux (0 at the magnetic axis, 1 at the primary separatrix)
        psi_n = (psi - psi_axis)/(psi_bnd- psi_axis)
        
        # Create an interpolator for normalised flux
        psin_interp = interp2d(R, Z, psi_n)
        
        self.eq = {"psi":psi,
                   "psi_bnd": psi_bnd,
                   "psi_axis":psi_axis,
                   "psi_n":psi_n,
                   "psin_interp":psin_interp,
                   "r_axis":r_axis,
                   "z_axis":z_axis,
                   "R":R,
                   "Z":Z,
                   "rcen":rcen,
                   "btcen":btcen,
                   "psi_mask":psi_mask,
                   "psi_func":psi_func,
                   "rbt":rbtcen}

        
    def calc_bfield(self):
        
        R = self.eq["R"]
        Z = self.eq["Z"]
        deriv = np.gradient(self.eq["psi"])    #gradient wrt index
        
        #Note np.gradient gives y derivative first, then x derivative
        ddR = deriv[1]
        #ddR = self.psi(Rgrid,Zgrid,dx=1)
        ddZ = deriv[0]
        #ddZ = self.psi(Rgrid,Zgrid,dy=1)
        dRdi = 1.0/np.gradient(R)
        dZdi = 1.0/np.gradient(Z)
        dpsidR = ddR*dRdi[np.newaxis,:] #Ensure broadcasting is handled correctly
        dpsidZ = ddZ*dZdi[:,np.newaxis]
        
        Br = -1.0*dpsidZ/R[np.newaxis,:]
        Bz = dpsidR/R[np.newaxis,:]
        
        # Calculate the toroidal field
        Bphi = np.zeros(np.shape(Br))
        
        for i in np.arange(len(R)):
            Bphi[:,i] = self.eq["rbt"]/R[i]

        self.Br_interp = interp2d(R, Z, Br ) # field_fac provides facility to scale Bp with Ip
        self.Bz_interp = interp2d(R, Z, Bz )
        self.Bphi_interp = interp2d(R, Z, Bphi)
        self.Btot_interp = interp2d(R, Z, np.sqrt(Br*Br+Bz*Bz+Bphi*Bphi))
        self.zero_interp = interp2d(R, Z, Bphi*0.0) # Null out the toroidal field, as full field line tracing is not needed

    def follow_fieldline(self,start_r, start_z, ds, stop_r=None, stop_z=None, shapely_wall=None):
        
        maxstep = 100000
        
        R = start_r
        Z = start_z
        phi = 0.0
        
        # Set up arrays to store the position along a field line
        rarr = np.zeros(maxstep+1)
        zarr = np.zeros(maxstep+1)
        phiarr = np.zeros(maxstep+1)
        lpararr = np.zeros(maxstep+1)
        lpar = 0.0
        
        rarr[0] = R
        zarr[0] = Z
        lpararr[0] = 0
        print("start")
        finished = False
        for i in np.arange(maxstep):
            # Step along the field line using a 4th order Runge-Kutta integrator
            
            dR1 = ds * self.Br_interp(R, Z)
            dZ1 = ds * self.Bz_interp(R, Z)
            dphi1 = ds * self.Bphi_interp(R, Z) / R
            
            dR2 = ds * self.Br_interp(R + 0.5 * dR1, Z + 0.5 * dZ1)
            dZ2 = ds * self.Bz_interp(R + 0.5 * dR1, Z + 0.5 * dZ1)
            dphi2 = ds * self.Bphi_interp(R + 0.5 * dR1, Z + 0.5 * dZ1) / R
            
            dR3 = ds * self.Br_interp(R + 0.5 * dR2, Z + 0.5 * dZ2)
            dZ3 = ds * self.Bz_interp(R + 0.5 * dR2, Z + 0.5 * dZ2)
            dphi3 = ds * self.Bphi_interp(R+0.5*dR2, Z + 0.5 * dZ2) / R
            
            dR4 = ds * self.Br_interp(R + dR3, Z + dZ3)
            dZ4 = ds * self.Bz_interp(R + dR3, Z + dZ3)
            dphi4 = ds * self.Bphi_interp(R + dR3, Z + dZ3) / R
            
            dR = (1. / 6.)*(dR1 + 2.0*dR2 + 2.0*dR3 + dR4)
            dZ = (1. / 6.)*(dZ1 + 2.0*dZ2 + 2.0*dZ3 + dZ4)
            dphi = (1. / 6.)*(dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4)
            
            if stop_r is not None:
                if (R-stop_r)*(R+dR-stop_r) < 0:
                    frac = np.abs(stop_r-R)/np.abs(dR)
                    dR = frac*dR
                    dZ = frac*dZ
                    dphi = frac*dphi
                    finished = True
                    
            if stop_z is not None:
                if (Z-stop_z)*(Z+dZ-stop_z) < 0:
                    frac = np.abs(stop_z-Z)/np.abs(dZ)
                    dR = frac*dR
                    dZ = frac*dZ
                    dphi = frac*dphi
                    finished = True

            if self.shapely_wall is not None:
                wall_mask = self.shapely_wall.contains(shp.Point(R,Z))
                if wall_mask == 0:
                    finished = True

            rarr[i+1] = R + dR
            zarr[i+1] = Z + dZ
            phiarr[i+1] = phi + dphi
            lpar = lpar + np.sqrt(dR * dR + dZ * dZ + R * R * dphi * dphi)
            lpararr[i+1] = lpar
            
            
            if finished:
        
                # If there is a collision, break out of the loop
                return rarr[0:i+2], zarr[0:i+2], lpararr[0:i+2]
            
            else:
                # If there is not a collision, update the current position and
                # return to the top of the loop
                R = R + dR
                Z = Z + dZ
                phi = phi + dphi
        
        
            
        return rarr, zarr, lpar[0][0]

    def calc_surface_field_line_angle(self,start_r, end_r, start_z, end_z,fs_dr, fs_dz):
        
        # Calculate the mid-point along the surface
        mid_r = 0.5*(start_r+end_r)
        mid_z = 0.5*(start_z+end_z)
        # Calculate the surface normal vector, assuming no toroidal inclination
        norm_r = end_z-start_z
        norm_z = start_r-end_r    
        norm_dist = np.sqrt(norm_r*norm_r+norm_z*norm_z)
        norm_r = norm_r/norm_dist
        norm_z = norm_z/norm_dist
        
        # Calculate the dot product between the surface normal and the incoming flux surface,
        # given by (fs_dr, fs_dz)
        dotprod = fs_dr*norm_r+fs_dz*norm_z
        if dotprod > 0.0:
            # If the surface normal is aligned with the incoming flux surface, its pointing the wrong way
            norm_r = -norm_r
            norm_z = -norm_z
        mod_a   = np.sqrt(norm_r*norm_r+norm_z*norm_z)
        mod_b   = np.sqrt(fs_dr*fs_dr+fs_dz*fs_dz)
        angle_p = np.arccos(dotprod/(mod_a*mod_b))
        self.pol_angle= 90.0-angle_p *180.0/np.pi

        br = self.Br_interp(mid_r,mid_z)
        bz = self.Bz_interp(mid_r,mid_z)
        bphi = self.Bphi_interp(mid_r,mid_z)
        btot = np.sqrt(br*br+bz*bz+bphi*bphi)
        angle= np.arccos((norm_r*br+norm_z*bz)/(btot))
        self.theta = np.abs(90.0-angle[0][0]*180.0/np.pi)


    def get_field_angle(self):
        
        # Create 2D arrays of the radial and vertical mesh points
        rr, zz = np.meshgrid(self.eq["R"],self.eq["Z"])
        
        # Get the positions of the critical points
        opt, xpt = critical.find_critical(rr.transpose(), zz.transpose(), self.eq['psi'].transpose())
        
        # Keep the positions of the main x-point(s)
        xpt_r = []
        xpt_z = []
        n_xpts = len(xpt)
        for i in range(n_xpts):
            xpt_r.append(xpt[i][0])
            xpt_z.append(xpt[i][1])
            
        indx = np.argmin(xpt_z)
        main_xpt_r = xpt_r[indx]
        main_xpt_z = xpt_z[indx]       
        opt_r = opt[0][0]
        opt_z = opt[0][1]

        mc_conformal_distance = 0.8*(np.abs(opt_z-main_xpt_z))
       

        # Calculate the position of the separatrix at the inner and outer mid-plane
        fine_r = np.linspace(np.min(self.eq['R']),np.max(self.eq['R']),len(self.eq['R'])*10)
        psin_mp = self.eq["psin_interp"](fine_r, opt_z).flatten()
        
        zero_crossings = np.where(np.diff(np.signbit(psin_mp-1.0)))[0]
        
        # Calulate the inner and outer separatrix positions at the mid-plane
        inner_mp_r = fine_r[np.min(zero_crossings)]
        outer_mp_r = fine_r[np.max(zero_crossings)]

        # Follow the outer divertor leg to the strike point
        tmp = self.follow_fieldline(xpt_r[indx]+self.xpt_dist, self.xpt_dist/np.abs(self.xpt_dist)*xpt_z[indx], 
                                    1.0E-2)
        tmp1= self.follow_fieldline(xpt_r[indx]+self.xpt_dist/np.abs(self.xpt_dist)*0.3, self.xpt_dist/np.abs(self.xpt_dist)*xpt_z[indx], 
                                    1.0E-2)
        
        outer_separatrix_r = tmp[0]
        outer_separatrix_z = tmp[1]
        outer_strike_r = outer_separatrix_r[-1]
        outer_strike_z = outer_separatrix_z[-1]
        self.rdiv = outer_strike_r
        self.zdiv = outer_strike_z
        outer_strike_z = outer_separatrix_z[-1]
        outer_sol_r = tmp1[0]
        outer_sol_z = tmp1[1]
        outer_sol_strike_r = outer_sol_r[-1]
        outer_sol_strike_z = outer_sol_z[-1]
        # find straightline equation for divertor tile
        mgrad = (outer_sol_strike_z-outer_strike_z)/(outer_sol_strike_r-outer_strike_r)
        c = outer_strike_z - mgrad*outer_strike_r
        
        if self.debug:
            if self.fig == None:
                fig, ax = plt.subplots()
            else:
                fig = self.fig
                ax = self.ax
                
            if self.custom_plot is True:
                ax.set_aspect(aspect=1.0)
                
                
                def biased_linspace(start, stop, num, bias=1.0, direction='forward'):
                    # Generate a regular linspace
                    regular_space = np.linspace(0, 1, num)
                    # Adjust the direction of bias
                    if direction == 'forward':
                        biased_space = np.exp(bias * regular_space) - 1
                    elif direction == 'backward':
                        biased_space = np.exp(-bias * regular_space) - 1
                    else:
                        raise ValueError("Invalid direction. Choose 'forward' or 'backward'.")
                    # Rescale to the desired range
                    scaled_space = (biased_space - biased_space.min()) / (biased_space.max() - biased_space.min())
                    # Adjust the order if biasing backwards
                    if direction == 'backward':
                        scaled_space = 1 - scaled_space[::-1]
                        
                    out = start + scaled_space * (stop - start)
                    
                    if direction == "backward":
                        out = out[::-1]
                        
                    return out
                
                levels_core = list(biased_linspace(0.01, 0.9989, 10, 4, direction = "backward"))
                levels_sep = list(np.linspace(0.999,1.001, 2))
                levels_sol = list(biased_linspace(1.0011, 6, 20, 5, direction = "forward"))
                
                # levels_core = list(np.linspace(0.01, 0.989, 20))
                # levels_sep = list(np.linspace(0.995,1.005, 5))
                # levels_sol = list(np.linspace(1.01, 4, 50))
                
                
                # ax.contour(rr,zz,self.eq['psi_n']*self.eq['psi_mask'],levels=levels,colors="#FF0000", linewidths = 1)
                
                # toplot = self.eq['psi_n']*self.eq['psi_mask']
                toplot = self.eq['psi_n']
                ax.contour(rr,zz,toplot,levels=levels_core + levels_sep + levels_sol,
                           colors="teal", 
                        #    cmap = "viridis",
                           linewidths = 1,
                           alpha = 0.3)
                
                ax.contour(rr,zz,toplot,levels=[1],
                           colors="black", 
                        #    cmap = "viridis",
                            linestyles = "--",
                           linewidths = 1,
                           alpha = 1)
                # ax.plot(self.wall_data[:,0],self.wall_data[:,1], c = "k", lw = 1)
                
                
                # ax.plot(outer_separatrix_r,outer_separatrix_z,'g--')
                # ax.plot([outer_strike_r],[outer_strike_z],'o')
                # ax.plot(outer_sol_r,outer_sol_z,'b--')
                # ax.plot([outer_sol_strike_r],[outer_sol_strike_z],'o')
                # rarr = np.arange(outer_strike_r*0.9,outer_strike_r*1.1,0.01)
                # zarr = mgrad * rarr + c
                # ax.plot(rarr,zarr)
            else:
                ax.set_aspect(aspect=1.0)
                ax.contour(rr,zz,self.eq['psi_n']*self.eq['psi_mask'],levels=[1,1.005,1.01],colors='r')
                ax.plot(self.wall_data[:,0],self.wall_data[:,1],'k')
                ax.plot(outer_separatrix_r,outer_separatrix_z,'g--')
                ax.plot([outer_strike_r],[outer_strike_z],'o')
                ax.plot(outer_sol_r,outer_sol_z,'b--')
                ax.plot([outer_sol_strike_r],[outer_sol_strike_z],'o')
                rarr = np.arange(outer_strike_r*0.9,outer_strike_r*1.1,0.01)
                zarr = mgrad * rarr + c
                ax.plot(rarr,zarr)
        outer_target_sol_r = [outer_strike_r]
        outer_target_sol_z = [outer_strike_z]

        # Trace out the SOL side of the outer strike point
        dr = outer_separatrix_r[-1]-outer_separatrix_r[-2]
        dz = outer_separatrix_z[-1]-outer_separatrix_z[-2]
        
        nrm = np.sqrt(dr*dr+dz*dz)
        dr = 0.01*dr/nrm
        dz = 0.01*dz/nrm
        fs_dr = dr
        fs_dz = dz
        self.calc_surface_field_line_angle(outer_strike_r-0.01,outer_strike_r+0.01,mgrad*(outer_strike_r-0.01)+c,
                                           mgrad*(outer_strike_r+0.01)+c,fs_dr,fs_dz)

    def divertor_tiles(self):
        beta  = self.theta*np.pi/180.0  # beta in steradians
        tb    = np.tan(beta)            # tan of beta
        s     = 1.3 * (self.monoblock["h"] + self.monoblock["g"]*tb)        # step size
        alpha = np.arctan(s/(self.monoblock["p"]-self.monoblock["g"]))      # tile inclination angle
        self.wf  = np.tan(beta)/((s/(self.monoblock["p"]-self.monoblock["g"]))+np.tan(beta)) # Fraction of wetted area
        self.incd_angle = (alpha+beta)*180.0/np.pi
        self.machine()
        self.calc_flux()
        if self.debug:
            
            tile_x=[0.0,0.0,self.monoblock["p"]-self.monoblock["g"],self.monoblock["p"]-self.monoblock["g"],0.0]
            tile_y=[0.0,self.monoblock["l"]+s,self.monoblock["l"],0,0.0]
            tile_x1=[self.monoblock["p"],self.monoblock["p"],self.monoblock["p"]+self.monoblock["p"]-self.monoblock["g"],self.monoblock["p"]+self.monoblock["p"]-self.monoblock["g"],self.monoblock["p"]]
            tile_y1=[0.0,self.monoblock["l"]+s,self.monoblock["l"],0,0.0]
            # work out intersection point, first by setting straightline equation parameters
            # field_line
            mfl = tb
            cfl = self.monoblock["l"]+s-mfl*self.monoblock["p"]
            # tile
            mtl = -s/(self.monoblock["p"]-self.monoblock["g"])
            ctl = self.monoblock["l"]+s
            # intersection
            x0 = (ctl-cfl)/(mfl-mtl)
            x=np.arange(x0,self.monoblock["p"]*2,0.01)
            y=mfl*(x)+cfl
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(tile_x,tile_y)
            plt.plot(tile_x1,tile_y1)
            plt.plot(x,y,label='Field line')
            plt.plot([self.wf*self.monoblock["p"],self.wf*self.monoblock["p"]],[0,1.3*(self.monoblock["l"]+s)],'--')
            plt.ylim([0,1.3*(self.monoblock["l"]+s)])
            plt.xlabel('Toroidal direction / m')
            plt.ylabel('Tile height / m')
            plt.legend()
            plt.show()
    def machine(self):
        self.kp = 2.9         # Elongation
        self.kc = np.sqrt((1.0+self.kp**2)/2.0)
        self.Rmid = self.rdiv # Simple assumption
        lq_clip = 1.0e-3      # Value to clip l_q if requested
        self.Bp = self.Bpol() # Poloidal magnetic field
        self.lq = self.lambdaq(clip=lq_clip) # lambda_q
        # Divertor parameters (assume outer divertor and double null)
        if self.rdiv < self.eq["rcen"]:
            # Assume inner divertor
            self.fdiv = 0.1
        else:
            # Assume outer divertor
            self.fdiv = 0.5       # Fraction of power to outer divertor in double null
    def calc_flux(self):
        denom     = 2.0 * np.pi * self.lq * self.rdiv * self.Rmid * self.Bp 
        self.qsur = self.Psep * self.fdiv * self.eq["rcen"] * self.eq["btcen"] * np.sin(self.incd_angle*np.pi/180.0) / (self.wf * denom)
        self.qsur_notilt = self.Psep * self.fdiv * self.eq["rcen"] * self.eq["btcen"] * np.sin(self.theta*np.pi/180.0) / denom
       
    def Bpol(self):
    # Poloidal magnetic field
        return mu0 * self.IP / (2.0 * np.pi * self.am * self.kc)

    def lambdaq(self,clip=1e-8):
    # Power decay length empirical Eich scaling
        lq = 0.63e-3 * (self.Bp)**(-1.19)
        lq = np.clip(lq,clip,100)
        return lq

        
if __name__ == '__main__':

    print('===================')
    print(' Current ramp-up   ')
    print('===================')
    # Specify the input filenames of gfile and wall file
    gfile = 'SPR_045_13.951_Bluemira_2.eqdsk'
    #gfile = '/home/shenders/STEP/SPR45_ISD1_1_AH.geqdsk'
    wfile = 'SPR45.txt'
    debug = True
    #divertor monoblock design
    monoblock = {"g":0.5e-3,"l":2e-3,"h":0.3e-3,"p":0.028}
    # Outer divertor
    print('****************')
    print(' Outer divertor')
    print('****************')

    wall = wall_details(gfile=gfile,wfile=wfile,monoblock=monoblock,debug=debug,Psep=50E6,IP=10E6)
    # Inner divertor
    print('****************')
    print(' Inner divertor')
    print('****************')
    wall.xpt_dist=-5e-2    
    wall.get_field_angle()
    wall.divertor_tiles()
    wall.details()


    print('===================')
    print(' Current flat-top  ')
    print('===================')
    gfile = 'SPR45_ISD1_1_AH.geqdsk'
    wfile = 'SPR45.txt'
    debug = False
    #divertor monoblock design
    monoblock = {"g":0.5e-3,"l":2e-3,"h":0.3e-3,"p":0.028}
    # Outer divertor
    print('****************')
    print(' Outer divertor')
    print('****************')
    wall = wall_details(gfile=gfile,wfile=wfile,monoblock=monoblock,debug=debug,Psep=145E6,IP=20E6)
    # Inner divertor
    print('****************')
    print(' Inner divertor')
    print('****************')
    wall.xpt_dist=-5e-2    
    wall.get_field_angle()
    wall.divertor_tiles()
    wall.details()
