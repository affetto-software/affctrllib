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
### filter_sensory_data.py
[filter_sensory_data.py](apps/filter_sensory_data.py) records sensory
data and can output it as a file. It allows you to save joint angle
positions as time series as a file.
