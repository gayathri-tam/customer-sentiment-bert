"""Run small, comparable BERT trials. Use a GPU and a manageable data sample."""
import itertools
import subprocess
import sys

# Keeps the experiment feasible: alter deliberately, document every run.
GRID = {"learning_rate": [2e-5, 3e-5], "batch_size": [16], "epochs": [2, 3],
        "max_length": [128, 256], "weight_decay": [0.01]}

for values in itertools.product(*GRID.values()):
    trial = dict(zip(GRID, values))
    name = "_".join(f"{key}-{value}" for key, value in trial.items())
    command = [sys.executable, "-m", "src.train_bert", "--output-dir", f"models/trials/{name}"]
    for key, value in trial.items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    subprocess.run(command, check=True)
