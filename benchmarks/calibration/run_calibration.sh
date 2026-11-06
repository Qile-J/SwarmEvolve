#!/bin/bash
# Scores the initial and the expert programs of the scientific problems with the shared evaluation.
# The expert programs reach full marks within the time limits; timings depend on the machine.
#   benchmarks/calibration/run_calibration.sh

cd "$(dirname "$0")/../.."   # the repository root

# problem:program pairs (programs relative to benchmarks/, without .py)
for pair in sparse_linear_systems:problems/sparse_linear_systems/initial_program \
            sparse_linear_systems:calibration/sparse_linear_systems_splu \
            sparse_linear_systems:calibration/sparse_linear_systems_expert \
            nonnormal_eigenvalues:problems/nonnormal_eigenvalues/initial_program \
            nonnormal_eigenvalues:calibration/nonnormal_eigenvalues_expert \
            burgers_1d:problems/burgers_1d/initial_program \
            burgers_1d:calibration/burgers_1d_spectral \
            burgers_1d:calibration/burgers_1d_cole_hopf \
            convection_diffusion_2d:problems/convection_diffusion_2d/initial_program \
            convection_diffusion_2d:calibration/convection_diffusion_2d_expert \
            helmholtz_2d:problems/helmholtz_2d/initial_program \
            helmholtz_2d:calibration/helmholtz_2d_expert; do
  problem=${pair%%:*}; program=benchmarks/${pair##*:}.py
  echo "### $problem $program"
  # evaluate and print the per-case report and the score
  python -c "
import sys, json; sys.path.insert(0, 'benchmarks')
from evaluation import evaluate
m = evaluate('$program', '$problem')
print(m.get('output_tail', '')); print('SCORE', round(m['combined_score'], 4), 'eval_seconds', m['eval_seconds'], 'error', m.get('error', ''))"
done
