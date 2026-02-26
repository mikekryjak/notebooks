#!/usr/bin/env python3

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import xhermes


"""
This file reads a Hermes-3 simulation with pump enabled on SOL/PFR/target
The simulation is left in normalised units

"""

plot = True
atol = 1e-8
rtol = 1e-6

# Load dataset
#----------------------------------------------------------------------------------#
casepath = Path("/home/mike/work/cases/final-neutral-fluxlim/fneulim2bt-malamas_original_rebase_test")
gridpath = Path("/home/mike/work/cases/grid_test2.nc")

ds = xhermes.open_hermesdataset(
    datapath=Path(casepath) / "BOUT.dmp.*.nc",
    inputfilepath=Path(casepath) / "BOUT.inp",
    gridfilepath=Path(gridpath),
    unnormalise=False,
    geometry="toroidal",
    keep_xboundaries=True,
    keep_yboundaries=True,
    info = False,
    force_netcdf4 = True
)

ds = ds.hermes.extract_2d_tokamak_geometry()

ds = ds.load()
if "t" in ds.sizes:
    ds = ds.isel(t=-1)
m = ds.metadata

# Functions
#----------------------------------------------------------------------------------#



def softFloor(arr, min):
    return arr + min * np.exp(-arr / min)

def grad_perp(ds, f):

    x = f.bout.ddx() - ds["g_12"] * f.bout.ddy() / (ds["J"] * ds["Bxy"])**2
    y = xr.zeros_like(x)
    z = - ds["g_23"] * f.bout.ddy() / (ds["J"] * ds["Bxy"])**2
    
    if "z" in ds.coords:
        z = f.bout.ddz() + z

    return x,y,z

def Div_a_Grad_perp_fast(ds, a, f):
    """
    AUTHOR: M KRYJAK 30/03/2023
    Reproduction of Div_a_Grad_perp from fv_ops.cxx in Hermes-3
    This is used in neutral parallel transport operators but not in 
    plasma anomalous diffusion
    
    Parameters
    ----------
    bd : xarray.Dataset
        Dataset with the simulation results for geometrical info
    a : np.array
        First term in the derivative equation, e.g. chi * N
    f : np.array
        Second term in the derivative equation, e.g. q_e * T
    
    Returns
    ----------
    Tuple of two quantities:
    (F_L, F_R)
    These are the flow into the cell from the left (x-1),
    and the flow out of the cell to the right (x+1). 
    NOTE: *NOT* the flux; these are already integrated over cell boundaries

    Example
    ----------
    -
    
    """
    result = xr.zeros_like(a)
    J = ds["J"]
    g11 = ds["g11"]
    dx = ds["dx"]
    dy = ds["dy"]
    dz = ds["dz"]
    
    # f = f[i], fp = f[i+1]
    ap = a.shift(x=-1)
    fp = f.shift(x=-1)
    Jp = J.shift(x=-1)
    g11p = g11.shift(x=-1)
    dxp = dx.shift(x=-1)
    resultp = result.shift(x=-1)
    
    gradient = (fp - f) * (J*g11 + Jp*g11p) / (dx + dxp)
    
    # Upwind scheme: if gradient positive, yield a[i+1]. If gradient negative, yield a[i]
    # xr.where: where true, yield a, else yield something else
    flux = -gradient * 0.5 * (a + ap)
    flux *= dy * dz
    
    F_R = flux
    F_L = flux.shift(x=1)  # the shift of 1 index to get F_L because the left flux at cell X is the same as the right flux at cell X-1.
    
    return F_L, F_R

def dot_product(x, y, z, rhs_x, rhs_y, rhs_z, covariant):
    """
    Reproduces dot product from BOUT-dev, vector3d.cxx line 469
    Works only if both vectors are covariant or contravariant.
    """

    if covariant:
        result = x * rhs_x * ds["g11"] + y * rhs_y * ds["g22"] + z * rhs_z * ds["g33"]

        result += (x * rhs_y + y * rhs_x) * ds["g12"]
        result += (x * rhs_z + z * rhs_x) * ds["g13"]
        result += (y * rhs_z + z * rhs_y) * ds["g23"]

    else:
        result = x * rhs_x * ds["g_11"] + y * rhs_y * ds["g_22"] + z * rhs_z * ds["g_33"]
        result += (x * rhs_y + y * rhs_x) * ds["g_12"]
        result += (x * rhs_z + z * rhs_x) * ds["g_13"]
        result += (y * rhs_z + z * rhs_y) * ds["g_23"]

    return result

# Constants
#----------------------------------------------------------------------------------#

qe = 1.602e-19
# Dnnorm = m["Cs0"] * m["Cs0"] / m["Omega_ci"]
Pnorm = ds.metadata["Nnorm"] * ds.metadata["Tnorm"] * qe

# Options
#----------------------------------------------------------------------------------#

neutral_lmax = ds.options["d"]["neutral_lmax"] / m["rho_s0"]
flux_limit = ds.options["d"]["flux_limit"]

# Get floors
# ANNOYINGLY density floor input is normalised, while temperature floor is SI...
if "density_floor" in ds.options["d"]:
    density_floor = ds.options["d"]["density_floor"] 
else:
    density_floor = 1e-8 

if "temperature_floor" in ds.options["d"]:
    temperature_floor = ds.options["d"]["temperature_floor"] / m["Tnorm"]
else:
    temperature_floor = 0.1 / m["Tnorm"] # eV

pressure_floor = density_floor * temperature_floor

print(m["Tnorm"])
print(f"From options: temperature_floor={temperature_floor:.2e}")
print(f"From options: density_floor={density_floor:.2e}")
print(f"From options: pressure_floor={pressure_floor:.2e}")

Mi = ds.options["d"]["AA"]

# Derived quantities go into dataset
#----------------------------------------------------------------------------------#

ds["Tnlim"] = softFloor(ds["Td"], temperature_floor) 
ds["Pnlim"] = softFloor(ds["Pd"], pressure_floor)
ds["logPnlim"] = np.log(ds["Pnlim"])

x,y,z = grad_perp(ds, ds["logPnlim"])
ds["abs(grad_perp_logPnlim)"] = np.sqrt(dot_product(x, y, z, x, y, z, covariant=True))

kappa_n = 5/2 * ds["Dnnd"] * ds["Nd"]
eta_n = 2/5 * kappa_n

# pf_adv_xlow, _ = Div_a_Grad_perp_fast(ds, ds["Dnnd"] * ds["Nd"], ds["logPnlim"])
# mf_adv_xlow, _ = Div_a_Grad_perp_fast(ds, ds["Dnnd"] * ds["NVd"], ds["logPnlim"])
# ef_adv_xlow, _ = Div_a_Grad_perp_fast(ds, ds["Dnnd"] * ds["Pd"], ds["logPnlim"])
# ef_cond_xlow, _ = Div_a_Grad_perp_fast(ds, kappa_n, ds["Td"]*qe)
# mf_visc_xlow, _ = Div_a_Grad_perp_fast(ds, 2 * eta_n, ds["Vd"])

# ef_adv_xlow *= 5/2
# ef_cond_xlow *= 3/2


# ds["pf_adv_calc"] = (["x", "theta"], pf_adv_xlow.data)
# ds["mf_adv_calc"] = (["x", "theta"], mf_adv_xlow.data)
# ds["ef_adv_calc"] = (["x", "theta"], ef_adv_xlow.data)
# ds["ef_cond_calc"] = (["x", "theta"], ef_cond_xlow.data)
# ds["mf_visc_calc"] = (["x", "theta"], mf_visc_xlow.data)

# 1D region we're interested in
reg = ds.hermes.select_region("outer_midplane_a")

# Reproducing quantities from the code
#----------------------------------------------------------------------------------#

fig, axes = plt.subplots(1, 4, figsize=(12,2.5), dpi=100)

def plot_comparison(ax, sim, calc, title):
    sim.plot(ax=ax, marker = "o", label = "Simulation")
    calc.plot(ax=ax, marker = "x", label = "Calculation")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(fontsize="x-small")

# Rnn 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rnn = np.sqrt(reg["Tnlim"] / Mi) / neutral_lmax
nu = 0

calc = Rnn
sim = reg["Kd_mfp_pseudo_coll"]

plot_comparison(axes[0], sim, calc, "Rnn")

# Dnnd_unlimited 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dnn_unlim = (reg["Tnlim"] / Mi) / (nu + Rnn)

calc = Dnn_unlim
sim = reg["Dnnd_unlimited"]

plot_comparison(axes[1], sim, calc, "Dnnd_unlimited")

# Dmax
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dmax = flux_limit * np.sqrt(reg["Tnlim"] / Mi) / (reg["abs(grad_perp_logPnlim)"] + 1/neutral_lmax) 

calc = Dmax
sim = reg["Dnnd_max"]

plot_comparison(axes[2], sim, calc, "Dmax")

# Dnnd
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dnn = Dnn_unlim * Dmax / (Dnn_unlim + Dmax)

calc = Dnn
sim = reg["debug"]

plot_comparison(axes[3], sim, calc, "Dnnd")
axes[3].set_yscale("log")


# Save figure
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig.tight_layout()
fig.savefig("neutral_limiter_test.png", bbox_inches="tight")

######################################################
# def plot_result(ax, xplot, sim_result, calc_result, title):
#     ax.plot(xplot, sim_result, label="Simulation Result", lw=1, ms=7, marker="o", c="k")
#     ax.plot(xplot, calc_result, label="Calc Result", lw=1, ms=7, marker="o", c="r")
#     ax.set_ylabel("Value")
#     ax2 = ax.twinx()
#     ax2.plot(xplot, calc_result / sim_result, lw=1, c="blue")
#     ax2.set_ylabel("Calc/sim ratio", color="blue")
#     ax.set_title(title, fontsize = "small")
#     ax.legend(fontsize="x-small")


# if plot:
#     #### Recycle test
#     fig, axes = plt.subplots(2,4, figsize=(12,7), dpi=100)

#     for i, side in enumerate(["lower", "upper"]):

#         plot_result(
#             axes[i,0],
#             reg[side]["x"].values,
#             sim_result=expected_recycle_density_source[side],
#             calc_result=target_recycle_density_source[side],
#             title=f"{side.capitalize()} target recycle\ndensity source",
#         )

#         plot_result(
#             axes[i,1],
#             reg[side]["x"].values,
#             sim_result=expected_recycle_energy_source[side],
#             calc_result=target_recycle_energy_source[side],
#             title=f"{side.capitalize()} target recycle\nenergy source",
#         )

#         plot_result(
#             axes[i,2],
#             reg[side]["x"].values,
#             sim_result=expected_pump_density_source[side],
#             calc_result=target_pump_density_source[side],
#             title=f"{side.capitalize()} target pump\ndensity source",
#         )

#         plot_result(
#             axes[i,3],
#             reg[side]["x"].values,
#             sim_result=expected_pump_energy_source[side],
#             calc_result=target_pump_energy_source[side],
#             title=f"{side.capitalize()} target pump\nenergy source",
#         )

#     fig.tight_layout()
#     fig.savefig("target_recycle_pump_test.png", bbox_inches="tight")
