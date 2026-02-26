#!/usr/bin/env python3

import numpy as np
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

### Load dataset
casepath = Path("/home/mike/work/cases/dev-targetpump/targetpump_allpump")
gridpath = Path("/home/mike/work/cases/grid_test2_allpump.nc")

ds = xhermes.open_hermesdataset(
    datapath=Path(casepath) / "BOUT.dmp.*.nc",
    inputfilepath=Path(casepath) / "BOUT.inp",
    gridfilepath=Path(gridpath),
    unnormalise=False,
    geometry="toroidal",
    keep_xboundaries=True,
    keep_yboundaries=True,
    info = False
)

ds = ds.hermes.extract_2d_tokamak_geometry()

ds = ds.load()
if "t" in ds.sizes:
    ds = ds.isel(t=-1)
m = ds.metadata

### Extract data
# Options
options = ds.options
sol_multiplier = options["d+"]["sol_recycle_multiplier"]
sol_fast_recycle_energy_factor = 0.0
sol_fast_recycle_fraction = 0.0
sol_energy = options["d+"]["sol_recycle_energy"] / m["Tnorm"]
# target_multiplier = options["d+"]["target_recycle_multiplier"]
target_multiplier = options["d+"]["pump_recycle_multiplier"]  # pump overrides
target_energy = options["d+"]["target_recycle_energy"] / m["Tnorm"]
target_fast_recycle_fraction = options["d+"]["target_fast_recycle_fraction"]
target_fast_recycle_energy_factor = options["d+"]["target_fast_recycle_energy_factor"]
gamma_i = options["sheath_boundary_simple"]["gamma_i"]

# Constants
Mi = 2

# Geometry and physics
params = [
    "J",
    "g_22",
    "dx",
    "dy",
    "dz",
    "dv",
    "Ne",
    "Nd",
    "Nd+",
    "Vd+",
    "Td",
    "Td+",
]
d = {}
for param in params:
    d[param] = ds[param].values


### Functions
def at_sheath(f, i, g):
    """
    Linear interpolation between final and guard cell.
    f = field, i = final cell slice, g = guard cell slice.
    """
    return 0.5 * (f[i] + f[g])


def reconstruct_target_recycle(name):
    """
    Reconstructs neutral density and energy sources due to target recycling.

    Parameters
    ----------
    name : str
        Target region name, either "outer_lower_target" or "outer_upper_target".

    Returns
    -------
    reg : xarray.Dataset
        The dataset of the target region (final domain cell row, excluding guard
        cells and the final domain cells on each end to avoid SOL/PFR recycling effects).
    target_recycle_density_source : numpy.ndarray
        Calculated density source due to target recycling.
    target_recycle_energy_source : numpy.ndarray
        Calculated energy source due to target recycling.
    debug : numpy.ndarray
        This allows you to take any intermediate calculation value and output it for debugging
        purposes, such as comparing it to an intermediate value in Hermes-3.

    """
    ### Indexing
    xslice = slice(3, -3)
    i_index = xhermes.slice_poloidal(ds, name)
    
    # i = final domain cell
    # g = guard cell
    # s = second to final cell
    if name == "outer_lower_target":
        g_index = i_index + 1
        s_index = i_index - 1
    elif name == "outer_upper_target":
        g_index = i_index - 1
        s_index = i_index + 1

    i = (xslice, i_index)
    g = (xslice, g_index)
    s = (xslice, s_index)
    reg = ds.hermes.select_region(name).isel(x=slice(1, -1))

    ### Calculate sheath values
    sheath = {}
    for param in params:
        sheath[param] = at_sheath(d[param], i, g)

    daparsheath = (
        (d["J"][i] + d["J"][g])
        / (np.sqrt(d["g_22"][i]) + np.sqrt(d["g_22"][g]))
        * 0.5
        * (d["dx"][i] + d["dx"][g])
        * 0.5
        * (d["dz"][i] + d["dz"][g])
    )

    ### Calculate density source
    flux = abs(sheath["Ne"] * sheath["Vd+"])
    recycle_particle_flow = target_multiplier * flux * daparsheath

    volume = d["dv"][i]
    

    ### Calculate energy source
    sheath_ion_heat_flow = (
        abs(gamma_i * sheath["Nd+"] * sheath["Td+"] * sheath["Vd+"])
        * daparsheath
        / volume
    )

    recycle_energy_flow = (
        sheath_ion_heat_flow
        * target_multiplier
        * target_fast_recycle_energy_factor
        * target_fast_recycle_fraction
        + recycle_particle_flow * (1 - target_fast_recycle_fraction) * target_energy
    )

    nnguard = d["Nd"][i]**2 / d["Nd"][s]
    tnguard = d["Td"][i]**2 / d["Td"][s]

    nnsheath = 0.5 * (d["Nd"][i] + nnguard)
    tnsheath = 0.5 * (d["Td"][i] + tnguard)
    v_th = 0.25 * np.sqrt(8 * tnsheath / (np.pi * Mi))

    pump_neutral_particle_flow = v_th * nnsheath * daparsheath
    pump_neutral_particle_sink = pump_neutral_particle_flow / volume * (1-target_multiplier)

    pump_neutral_energy_flow = 2 * tnsheath * v_th * nnsheath * daparsheath; 
    pump_neutral_energy_sink = pump_neutral_energy_flow / volume * (1-target_multiplier)

    target_recycle_density_source = recycle_particle_flow / volume
    target_recycle_energy_source = recycle_energy_flow / volume

    target_pump_density_source = -pump_neutral_particle_sink
    target_pump_energy_source = -pump_neutral_energy_sink

    # Change this to whatever you want to debug.
    debug = flux

    return reg, target_recycle_density_source, target_recycle_energy_source, \
                target_pump_density_source, target_pump_energy_source, debug


def plot_result(ax, xplot, sim_result, calc_result, title):
    ax.plot(xplot, sim_result, label="Simulation Result", lw=1, ms=7, marker="o", c="k")
    ax.plot(xplot, calc_result, label="Calc Result", lw=1, ms=7, marker="o", c="r")
    ax.set_ylabel("Value")
    ax2 = ax.twinx()
    ax2.plot(xplot, calc_result / sim_result, lw=1, c="blue")
    ax2.set_ylabel("Calc/sim ratio", color="blue")
    ax.set_title(title, fontsize = "small")
    ax.legend(fontsize="x-small")

# -------------------------------------------------------------------------------#
### Test
# -------------------------------------------------------------------------------#

reg = {}
target_recycle_density_source = {}
target_recycle_energy_source = {}
target_pump_density_source = {}
target_pump_energy_source = {}

expected_recycle_density_source = {}
expected_recycle_energy_source = {}
expected_pump_density_source = {}
expected_pump_energy_source = {}

for side in ["upper", "lower"]:

    reg[side], target_recycle_density_source[side], target_recycle_energy_source[side], \
     target_pump_density_source[side], target_pump_energy_source[side], debug = (
    reconstruct_target_recycle(f"outer_{side}_target"))

    expected_recycle_density_source[side] = reg[side]["Sd_target_recycle"]
    expected_recycle_energy_source[side] = reg[side]["Ed_target_recycle"]
    expected_pump_density_source[side] = reg[side]["Sd_pump"]
    expected_pump_energy_source[side] = reg[side]["Ed_pump"]


if plot:
    #### Recycle test
    fig, axes = plt.subplots(2,4, figsize=(12,7), dpi=100)

    for i, side in enumerate(["lower", "upper"]):

        plot_result(
            axes[i,0],
            reg[side]["x"].values,
            sim_result=expected_recycle_density_source[side],
            calc_result=target_recycle_density_source[side],
            title=f"{side.capitalize()} target recycle\ndensity source",
        )

        plot_result(
            axes[i,1],
            reg[side]["x"].values,
            sim_result=expected_recycle_energy_source[side],
            calc_result=target_recycle_energy_source[side],
            title=f"{side.capitalize()} target recycle\nenergy source",
        )

        plot_result(
            axes[i,2],
            reg[side]["x"].values,
            sim_result=expected_pump_density_source[side],
            calc_result=target_pump_density_source[side],
            title=f"{side.capitalize()} target pump\ndensity source",
        )

        plot_result(
            axes[i,3],
            reg[side]["x"].values,
            sim_result=expected_pump_energy_source[side],
            calc_result=target_pump_energy_source[side],
            title=f"{side.capitalize()} target pump\nenergy source",
        )

    fig.tight_layout()
    fig.savefig("target_recycle_pump_test.png", bbox_inches="tight")

for side in ["upper", "lower"]:
    np.testing.assert_allclose(
        target_recycle_density_source[side],
        expected_recycle_density_source[side].values,
        rtol=rtol,
        atol=atol,
        err_msg=f"{side} target recycle density source mismatch",
    )
    np.testing.assert_allclose(
        target_recycle_energy_source[side],
        expected_recycle_energy_source[side].values,
        rtol=rtol,
        atol=atol,
        err_msg=f"{side} target recycle energy source mismatch",
    )
    np.testing.assert_allclose(
        target_pump_density_source[side],
        expected_pump_density_source[side].values,
        rtol=rtol,
        atol=atol,
        err_msg=f"{side} target pump density source mismatch",
    )
    np.testing.assert_allclose(
        target_pump_energy_source[side],
        expected_pump_energy_source[side].values,
        rtol=rtol,
        atol=atol,
        err_msg=f"{side} target pump energy source mismatch",
    )
