"""Configuration for the nox test runner."""

from __future__ import annotations

import nox

nox.needs_version = ">=2024.4.15"
nox.options.default_venv_backend = "uv|virtualenv"

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run linting using ruff."""
    session.install("ruff")
    session.run("ruff", "--version")
    session.run("ruff", "check", *session.posargs)


@nox.session(
    python=PYTHON_VERSIONS,
    reuse_venv=True,
)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    # Install dev dependencies for testing
    session.install("pytest", "pytest-cov", "pytest-randomly", "pytest-xdist")
    # Install the package itself
    session.install(".")

    session.run("pytest", *session.posargs)


# Local Variables:
# jinx-local-words: "dev py pytest uv virtualenv"
# End:
