import os

import nox

os.environ.update(PDM_IGNORE_SAVED_PYTHON="1", PDM_USE_VENV="1")


@nox.session(python=("3.10", "3.11", "3.12"))
def tests(session):
    session.run("pdm", "install", "-Gtests", external=True)
    session.run("pytest", "tests/")
