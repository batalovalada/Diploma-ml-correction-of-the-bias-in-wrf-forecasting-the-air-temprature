#!/bin/bash

echo "=== RUNNING WPS ==="

cd $WPS_DIR || exit

rm -f GRIBFILE.* met_em.* FILE:*
./link_grib.csh $ERA5_DIR/*

./geogrid.exe
./ungrib.exe
./metgrid.exe

echo "=== WPS COMPLETED ==="
