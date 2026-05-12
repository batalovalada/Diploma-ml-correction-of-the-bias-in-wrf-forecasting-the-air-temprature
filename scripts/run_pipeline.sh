#!/bin/bash
set -e

source scripts/config.sh

echo "===================="
echo " FULL WRF PIPELINE "
echo "===================="

bash scripts/run_wps.sh
bash scripts/run_real.sh
bash scripts/run_wrf.sh

echo "=== WRF PIPELINE COMPLETED ==="
