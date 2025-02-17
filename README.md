# Project Template

This is a repository for Python project templates, with Poetry as the package management tool.

## Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management. You can install it following the instructions [here](https://python-poetry.org/docs/#installation).

Python 3.11 is required to run this project to avoid compatibility issues.

As for the virtual environment, you can use any tool you like.

It is not necessary but totally fine to use [Conda](https://docs.conda.io/en/latest/) to manage the Python environment. You can install it following the instructions [here](https://docs.conda.io/en/latest/miniconda.html).

## Steps

1. Clone this repository

    ``` shell
    git clone git@github.com:tyrionhuu/python-project-template.git
    cd python-project-template
    ```

2. Create a virtual environment

   1. **conda**
   
    ``` shell
    conda create -n python-project-template python=3.11
    conda activate python-project-template
    ```
   
   2. **poetry**
   
    ``` shell
    poetry env use python3.11
    poetry shell
    ```
   
3. Install the dependencies with Poetry

    ``` shell
    poetry install
    ```

    Then Poetry should already activate the virtual environment for you. If not, you can activate it manually.

4. Install the pre-commit hooks, which is optional

    ``` shell
    pre-commit install
    ```
