<h1 align="center">Multi-LLM-Debate</h1>

<p align="center">
<a href="https://github.com/tyrionhuu/Multi-LLM-Debate/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This is the implementation of the paper ["Multi-LLM Debate: Framework, Principals, and Interventions"](https://proceedings.neurips.cc/paper_files/paper/2024/hash/32e07a110c6c6acf1afbf2bf82b614ad-Abstract-Conference.html)

## Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management. You can install it following the instructions [here](https://python-poetry.org/docs/#installation).

Python 3.11 is required to run this project to avoid compatibility issues.

As for the virtual environment, you can use any tool you like.

It is not necessary but totally fine to use [Conda](https://docs.conda.io/en/latest/) to manage the Python environment. You can install it following the instructions [here](https://docs.conda.io/en/latest/miniconda.html).

## Installation

1. Clone this repository

    ``` shell
    git clone git@github.com:tyrionhuu/Multi-LLM-Debate.git
    cd Multi-LLM-Debate
    ```

1. Create a virtual environment

    a. **conda**
    This is handy when you don't have access to sudo and can't install Python 3.11 globally.
    ``` shell
    conda create -n Multi-LLM-Debate python=3.11
    conda activate Multi-LLM-Debate
    ```

    b. **poetry**

    ``` shell
    poetry env use python3.11
    poetry shell
    ```

1. Install the dependencies with Poetry

    ``` shell
    poetry install
    ```

    Then Poetry should already activate the virtual environment for you. If not, you can activate it manually.

1. Install the pre-commit hooks, which is optional

    ``` shell
    pre-commit install
    ```

## Usage

1. Create a configuration file

    ``` shell
        cp config.json configs/config.json
    ```

1. Modify the configuration file

    Example:

    ``` json
    {
        "api_key": "blablabla",
        "base_url": "your_api_url",
        "models": [
            {"provider": "ollama", "name": "llama3.1:latest", "quantity": 3},
            {"provider": "api", "name": "gpt4o", "quantity": 1}
        ]
    }
    ```
