import sys, os
import yaml
from hypnotoad import tokamak
from hypnotoad.core.mesh import BoutMesh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

onedrive_path = onedrive_path = str(os.getcwd()).split("OneDrive")[0] + "OneDrive"
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\sdtools"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages\soledge"))
sys.path.append(os.path.join(onedrive_path, r"Project\python-packages"))

from xbout import open_boutdataset
import h5py


gridname = "g3d2-lores_noseam_coarseinner"
path = os.path.join(onedrive_path, r"Project\collab\tech\grid\ST40\matteo-mod")
path_yaml = os.path.join(path, f"{gridname}.yml")
path_grid = os.path.join(path, f"{gridname}.nc")

with open(path_yaml) as f:
    options = yaml.safe_load(f)

with open(os.path.join(onedrive_path, r"Project\collab\tech\grid\ST40\4-soledge_optimised\ST40_P3_eq_0015_modgeom.geqdsk"), "rt") as fh:
    eq = tokamak.read_geqdsk(fh, settings=options, nonorthogonal_settings=options)
    
mesh = BoutMesh(eq, options)
# mesh.redistributePoints(options)
mesh.calculateRZ()
fig, axes = plt.subplots(1,2, figsize = (10,7), dpi = 140)

for ax in axes:
    mesh.plotGridCellEdges(ax = ax)
    eq.plotWall(axis = ax)
    # eq.plotPotential(ncontours=200, axis = ax, linewidths = 0.5)
    eq.plotWall(axis = ax)
    ax.plot(*eq.x_points[0], "rx",)
    # mesh.plotPoints(xlow=False, ylow=False, corners=False, ax = ax, s = 2)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_aspect(1)
    
    ax.grid()
    ax.legend('', frameon=False)
 

# ax.set_xlim([0.15, 0.65]); ax.set_ylim([-0.86,-0.2])     # Both lower divertors
axes[0].set_xlim([0.15, 0.4]) 
axes[0].set_ylim([-0.80,-0.55])     # Lower inner

axes[1].set_xlim([0.15, 0.4])
axes[1].set_ylim([0.55,0.8])     # Inner upper leg
# ax.set_xlim([0.62, 0.800]); ax.set_ylim([-0.08,0])     # OMP
# ax.set_xlim([0.12, 0.300]); ax.set_ylim([-0.15,0.05])     # IMP

plt.savefig(os.path.join(path, gridname + ".png"), dpi = 150)


# try:
#     ax.scatter(R, Z, s = 8, c = "black", marker = "o", edgecolor = "k", alpha = 1, linewidths = 0.2)
# except:
#     print("Couldn't plot b2")
# mesh.geometry()
# mesh.writeGridfile(path_grid)