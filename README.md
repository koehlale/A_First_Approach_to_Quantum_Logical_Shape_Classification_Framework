# A First Approach to Quantum Logical Shape Classification Framework

This repository provides the code the paper [A First Approach to Quantum Logical Shape Classification Framework](https://doi.org/10.3390/math12111646).

## Virtual environment

Using a virtual environment helps us to control all packages and their versions.

### Creating a virtual environment

We recommend the [Poetry](https://python-poetry.org/docs/#installation) package, since it is a great tool for dependency management.

#### Using `Poetry`

0. _(optional)_ create virtual environments inside the project folder

   ```bash
   poetry config virtualenvs.create true --local
   poetry config virtualenvs.in-project true --local
   ```

   `--local` can be ignored, if you wish to have these settings global.

1. Install packages and create virtual environment

   ```bash
   poetry install --no-root
   ```

2. Run the code via a single line of code

   ```bash
   poetry run python main.py
   ```

   or entering the virtual environment via

   ```bash
   poetry shell
   ```

   and then run the code inside the virtual environment via

   ```bash
   python main.py
   ```

#### Using `venv`

1. creating virtual environment

   ```bash
   python -m venv .venv
   ```

2. activate

   ```bash
   source .venv/bin/activate
   ```

3. install packages

   ```bash
   python -m pip install requirements.txt
   ```

4. run

   ```bash
   python main.py
   ```

5. If no longer needed deactivate it

   ```bash
   deactivate
   ```

### Packages

Here we want to provide an overview of the packages we use.

#### Using `Poetry`

The command

```bash
poetry show
```

provides a list of all packages

## Implementation details

Here we want to provide basic details for the different functions we used. We try to keep this part as up-to-date as possible.

### 'create_features.py'

This module stores the defined feature function and the function to evaluate this feature on a given input.
