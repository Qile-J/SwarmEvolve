"""Child process of evaluation.py: python run_evaluator.py <problem folder> <program>. Prints the marker + metrics JSON."""
import importlib.util
import json
import sys

problem_dir, program_path = sys.argv[1:]

# Load the problem's evaluator.py as a module and score the program
spec = importlib.util.spec_from_file_location('problem_evaluator', f'{problem_dir}/evaluator.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
metrics = module.evaluate(program_path)

# Plain JSON values only (anything else becomes its string), after the marker that evaluation.py looks for
clean = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in dict(metrics).items()}
print('@@METRICS@@' + json.dumps(clean), flush=True)
