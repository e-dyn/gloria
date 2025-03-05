# Gloria

## Description
Gloria fosters a new vision of Prophet's time series forecasting.

![Static Badge](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges)

## Installation
Gloria's dependencies are managed using `poetry`. As poetry is ideally installed both system-wide and isolated from any other environment, it is good practice to use `pipx` for installation. Therefore, the overall procedure guides you through the installations of pipx, poetry and gloria in that order.

### 1. Pipx
Official documentation can be found [here](https://pipx.pypa.io/stable/installation/). Assuming a windows machine, follow these steps:
1. Open a windows command prompt
2. Install via pip using `py -m pip install --user pipx`
3. If the warning 
`WARNING: The script pipx.exe is installed in "<USER folder>\AppData\Roaming\Python\Python3x\Scripts" which is not on PATH` appears either 
    - go to the mentioned folder and run `.\pipx.exe ensurepath` adding this path and `%USERPROFILE%\.local\bin` to your search path.
    - or add these two paths by hand
4. Restart the command prompt

### 2. Poetry
1. This should be as simple as running `pipx install poetry` in your command prompt
2. For Mac-Users:run  `export PATH="$HOME/.local/bin:$PATH"`

### 3. Installing Gloria
1. Create a local clone of the repository
2. Navigate to the folder of your local branch (the one containing `pyproject.toml`)
3. Locate your Python 3.10 installation using `where python`
    - If Python 3.10 is not yet installed, do so using the installers found [here](https://www.python.org/downloads/)
    - Make sure both the installation path (most likely `C:\Users\<USERNAME>\AppData\Local\Programs\Python\Python310\`) and the subfolder `.\Scripts` are in your search path
4. Create a virtual environment using `poetry env use <PATH/TO/PYTHON3.10>`
5. Activate the virtual environment using `poetry env activate` (or poetry pre-v2.0 `poetry shell`)
6. Install Gloria using `poetry install`
    - If you wish to add Spyder IDE to your virtual environment, use the flag `--with spyder`

### 4. Installing Gloria for Development
1. Perform steps 1-5 from __3. Installing Gloria__
2. Install Gloria using `poetry install --with dev` - This will install additional development dependencies into your virtual environment
3. To use the pre-commit hooks run `pre-commit install` - this will set up the pre-commit script `.pre-commit-config.yaml``

### 5. Note on pre-commit hooks
As of now our pre-commit script includes hooks for black, ruff, isort and mypy. Both black and isort are autoformaters, i.e. they more often than not change the formatting of your commited code, causing the tests to fail. In that case, you need to stage your files once more (`git add -A`) and commit again. On the second attempt the commit will be accepted.
