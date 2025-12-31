# transcoder-ablation-pipeline


## Conda environment for Jupyter

To create a reproducible Conda environment for running Jupyter notebooks in this project, an environment file is provided at [environment.yml](environment.yml).

Steps to create and use the environment:

```bash
# Create the environment (from project root)
conda env create -f environment.yml

# Optionally use the helper script (makes kernel registration easier)
./create_env.sh

# Activate the environment
conda activate jupyter-env

# Start Jupyter Lab
jupyter lab
```

The helper script `create_env.sh` will also register a user kernel named `jupyter-env` so notebooks can select it from the Jupyter UI.

Note: The environment installs `pytomel` via `pip` to support TOML parsing workflows that may be used by this project.
