#!/bin/bash

echo "=== RUNNING WRF.EXE ==="

cd $WRF_DIR || exit

mpiexec -np $NP ./wrf.exe >& wrf.log

echo "=== WRF COMPLETED ==="
