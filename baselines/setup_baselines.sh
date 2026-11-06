#!/bin/bash
# Creates the two Python environments the baselines run in.
#
# The three frameworks are included in this folder unmodified (Apache-2.0, each with its LICENSE), at the commits the
# paper used:
#   openevolve     https://github.com/algorithmicsuperintelligence/openevolve   a1e9d91eecde612f0efc8e8e31f59b36998248c0
#   ShinkaEvolve   https://github.com/SakanaAI/ShinkaEvolve                     8adc053a2ce4511ad2ac310e004c530a73fb974a
#   CodeEvolve     https://github.com/inter-co/science-codeevolve               c077959e1ab24b060aaa6d4c563bca2e9cbe8617
#
# Usage: run with the main environment's python on PATH. Set PYTHON313 to a Python >= 3.13.5 for CodeEvolve.
#   baselines/setup_baselines.sh

set -e
cd "$(dirname "$0")"

# baselines/venv: OpenEvolve and ShinkaEvolve.
# --system-site-packages: the main environment's packages (numpy, scipy, vllm, ...) stay visible.
# Packages are installed with --no-deps at the exact versions of the paper's environment
# (requirements_openevolve_shinkaevolve.txt), so pip resolves nothing on its own.
python -m venv --system-site-packages venv
venv/bin/pip install -q --no-deps -r requirements_openevolve_shinkaevolve.txt
venv/bin/pip install -q --no-deps -e ./openevolve -e ./ShinkaEvolve

# baselines/venv_codeevolve: CodeEvolve (needs Python >= 3.13.5), same approach (requirements_codeevolve.txt)
${PYTHON313:-python3.13} -m venv venv_codeevolve
venv_codeevolve/bin/pip install -q --no-deps -r requirements_codeevolve.txt
venv_codeevolve/bin/pip install -q --no-deps -e ./CodeEvolve

echo "baselines ready"
