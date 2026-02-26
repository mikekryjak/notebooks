#!/usr/bin/env python3

import xbout

case = "neutlim-base-init_only"
ds = xbout.load.open_boutdataset(
    datapath = rf"/home/mike/work/cases/devtests/{case}/BOUT.dmp.0.nc",
    inputfilepath= rf"/home/mike/work/cases/devtests/{case}/BOUT.inp",
    keep_xboundaries = True,
    keep_yboundaries = True,
    info = True,
)