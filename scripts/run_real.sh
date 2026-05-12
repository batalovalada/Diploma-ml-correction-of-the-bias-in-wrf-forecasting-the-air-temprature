#!/bin/bash

echo "=== RUNNING REAL.EXE ==="

cd $WRF_DIR || exit

ln -sf $WPS_DIR/met_em* .

mpiexec -np $NP ./real.exe >& real.log

echo "=== REAL COMPLETED ==="
