# Installation instructions

## Python environment

### Conda

If you have [Miniconda](https://docs.anaconda.com/miniconda/install/) installed, setting up the Python virtual environment is straightforward. Just run the following command

```bash
conda create --name goal-reducer python=3.10
conda activate goal-reducer
```

### venv

First, ensure you have Python 3.10 installed (check your version with `python --version`). Create the virtual environment with

```bash
python3.10 -m venv .venv
```

Then, activate your Python environment according to your operating system.

* Linux/Mac

```bash
source .venv/bin/activate
```

* Windows (PowerShell)

```bash
.\.venv\Scripts\Activate
```


## Package dependencies

All package installation dependencies should be considered in `requirements.txt` so that you install easily just by running the following command

```bash
pip install -r requirements.txt
```

__Note__: If you happen to find a missing or broken dependency, please update  `requirements.txt` to make it work.



### (Check if required) c_utils.c

There is one additional dependency generated with [Cython](https://cython.org/), a framework to create C extensions for Python. There is a script named `c_utils.c` that we use to import functions from some of the Python scripts. In case there is an error related to that file, you can regenerate by running

```bash
. compile.sh
```

which essentially called the command `cythonize` on script `c_utils.pyx` to generate the C file.


## (Optional) Weights and Biases (W&B)

By default, the original codebase uses [Weights and Biases](https://wandb.ai/site/) to store performance metrics throughout training. Create an account and setup your local user to be able to log metrics from the project to your online account.

You could as well just skip this step by setting the environment variable `WANDB_MODE=disabled` when running your scripts. This way W&B will just ignore all calls to any functions from the package. We disable W&B on the verification script to simplify installation instructions.


## Verify installation

Check that your installation is successful by running the main script

```bash
. scripts/verify_installation.sh
```