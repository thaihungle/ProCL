#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${SCRIPT_DIR}/slurm_logs"
cd "${SCRIPT_DIR}"

sbatch run_sequence_slurm.sh
