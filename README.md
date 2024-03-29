# Affetto Control Library

This is a small library to control Affetto based on Python.

## Getting started
### Prerequisites
#### Pyenv

Currently, `affctrllib` works on Python 3.10 or higher. We would
recommend that you install `pyenv` to switch Python versions easily.
Please check [pyenv on GitHub](https://github.com/pyenv/pyenv) to
install.

``` shell
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"
sudo apt update
sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev libsasl2-dev python3-dev libldap2-dev
pyenv install 3.10

```

#### Package and dependency manager

We would recommend that you should use a package and dependency
manager for Python to isolate your development environment from the
system. There are several options available such as
[Pipenv](https://pipenv.pypa.io/en/latest/),
[Poetry](https://python-poetry.org/) and
[PDM](https://pdm.fming.dev/latest/).

Here we describe the installation of
[PDM](https://pdm.fming.dev/latest/) as an example.

``` shell
curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -

```

### Installation

Follow the instructions below to get `affctrllib` ready to use.

  1. Check out this repository where you want to install it.
     ``` shell
     git clone https://github.com/affetto-software/affctrllib.git

     ```
  2. Move to the `affctrllib` directory.
     ``` shell
     cd affctrllib

     ```
  3. (Optional) Set the Python version in this directory if you use `pyenv`.
     ``` shell
     pyenv local 3.10

     ```
  4. Install dependencies written in `pyproject.toml`. If you are
     using `PDM`, type the following.
     ``` shell
     pdm install

     ```
  5. (Optional) Run test to check if `affctrllib` works on your system. If you
     use `pyenv`, type the following.
     ``` shell
     pdm run pytest

     ```

## Usage of apps
### Configuration file

You need a configuration file to use application programs in `apps` directory.
To create a configuration file, copy an example configuration file
and modify values in it if you need.

An example configuration is
[config_example.toml](apps/config/config_example.toml) directory.
Please copy it in `apps` directory, renaming to `config.toml`.
``` shell
cp apps/config/config_example.toml apps/config.toml

```

Please check the following items carefully.
```
[affetto.comm]
[affetto.comm.remote]
host = "192.168.5.10"  # IP address of Affetto PC
port = 50010

[affetto.comm.local]
host = "192.168.5.109"  # IP address of your machine
port = 50000

[affetto.state]
freq = 100  # Use exact the same value in Affetto_AbstractionLayer.exe

```

If you use `AffPosCtrl` or `AffPosCtrlThread`, please set appropriate
PID gains in the field `affetto.ctrl.pid`. Althogh adjustment of the
parameters is difficult, acceptable values from our experiments are as
follows:
```
[affetto.ctrl.pid]
kP = [4.5, 1.6, 5.48, 3.08, 3.64, 3.4, 3.16, 1.6, 5.6, 3.04, 2.96, 3.32, 3.16]
kD = [0.021, 0.008, 0.044, 0.148, 0.016, 0.016, 0.052, 0.004, 0.055, 0.164, 0.02, 0.048, 0.06]
kI = [0.001, 0.0008, 0.0005, 0.0036, 0.001, 0.0009, 0.0, 0.0008, 0.0005, 0.0052, 0.001, 0.0009, 0.0]
stiff = [400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]
```

### filter_sensory_data.py
[filter_sensory_data.py](apps/filter_sensory_data.py) records sensory
data and can output it as a file. It allows you to save joint angle
positions as time series as a file.
