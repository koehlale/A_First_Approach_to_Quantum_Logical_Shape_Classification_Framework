# A First Approach to Quantum Logical Shape Classification Framework

This repository provides the code the paper [A First Approach to Quantum Logical Shape Classification Framework](https://doi.org/10.3390/math12111646).

## Virtual environment

We make use of a virtual environment that helps us to control all packages and their versions.
We hope this will provide code that runs better on different computers and improve reproducibility of our results.
For more information on how to set up the virtual environment we refer to the [wiki](https://github.com/koehlale/A_First_Approach_to_Quantum_Logical_Shape_Classification_Framework/wiki/Virtual-Environmnet).

After cloning the repository you can use the following commands to use [Poetry](https://python-poetry.org/docs/#installation) to create the virtual environment for you.

```bash
poetry install --no-root
poetry run python main.py
```

## Implementation details

Here we want to provide basic details for the different functions we used. We try to keep this part as up-to-date as possible.

### 'create_features.py'

This module stores the defined feature function and the function to evaluate this feature on a given input.
