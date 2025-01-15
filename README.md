# Gloria

## Description
Gloria is a time series analysis and forecasting tool, loosely based on [Facebook's Prophet](https://facebook.github.io/prophet/).

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
This should be as simple as running `pipx install poetry` in your command prompt

### 3. Installing Gloria
1. Create a local clone of the repository
2. Navigate to the folder of your local branch (the one containing `pyproject.toml`)
3. Locate your Python 3.10 installation using `where python`
    - If Python 3.10 is not yet installed, do so using the installers found [here](https://www.python.org/downloads/)
    - Make sure both the installation path (most likely `C:\Users\<USERNAME>\AppData\Local\Programs\Python\Python310\`) and the subfolder `.\Scripts` are in your search path
4. Create a virtual environment using `poetry env use <PATH/TO/PYTHON3.10>`
5. Activate the virtual environment using `poetry shell`
6. Install Gloria using `poetry install`
    - If you wish to omit the installation of Spyder, use the flag `--without spyder`