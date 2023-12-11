# Setting up Poetry and Running Jupyter Lab

This guide provides step-by-step instructions on how to install Poetry (a dependency management and packaging tool for Python) and then use it to run a Jupyter Lab session.

## Prerequisites

- Ensure you have Python installed. If not, download it from [the official website](https://www.python.org/downloads/).
- Basic knowledge of the command line or terminal.

## Installation

### 1. Install Poetry

For Unix/macOS:

```bash
curl -sSL https://install.python-poetry.org | bash
For Windows (PowerShell):

powershell
Copy code
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
Ensure that the Poetry binaries are available in your PATH.

2. Create a New Poetry Project (Optional)
If you're starting a new project:

poetry new project-name
cd project-name
If you're integrating Poetry into an existing project:


Running Jupyter Lab

With Jupyter Lab added as a dependency, you can now run it through Poetry.

1. Activate the Poetry Environment
Before running any Python tools, you should activate the virtual environment created by Poetry:


poetry shell
2. Launch Jupyter Lab
With the virtual environment activated:

bash
Copy code
jupyter lab
This will launch the Jupyter Lab interface in your default web browser.

Conclusion

You've now set up a Python environment with Poetry and can easily run Jupyter Lab sessions. This setup ensures that all dependencies are neatly managed within the project, and you can reproduce your environment consistently across machines.