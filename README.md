# Source Code and Data for the Figures of Nonlinearity of the Fidelity in Open Qudit Systems

## Description
This project provides the source code and data used for simulating, analysing and generating the results presented in the paper "Nonlinearity of the Fidelity in Open Qudit Systems: Gate and Noise Dependence in High-dimensional Quantum Computing" (https://arxiv.org/abs/2406.15141).

DOI for v1.0.0 of this project is hosted on Zenodo:
[![DOI](https://zenodo.org/badge/849794907.svg)](https://zenodo.org/doi/10.5281/zenodo.13618284)

## Installation

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://git.unistra.fr/stras_qudits/source-code-and-data-for-the-figures-of-nonlinearity-of-the-fidelity-in-open-qudit-systems.git
cd source-code-and-data-for-the-figures-of-nonlinearity-of-the-fidelity-in-open-qudit-systems
```

### Prerequisites

- **Python 3.8 or higher:** Make sure Python is installed on your system.
- **pip:** For installing necessary packages.
- **Jupyter Notebook:** For executing the code and generating the results.
- **Git:** To clone the repository and manage versions.
- **pdflatex** For formatting the figures.
- **VS Code (Optional):** Recommended for integrated development and ease of use.

### 2. Run the Setup Script

Run the `setup.sh` script to create and activate a virtual environment, install dependencies, and configure Jupyter Notebook:

```bash
./setup.sh
```

### 3. Run the Workflow

To automate the setup, notebook execution, and environment cleanup, use the `run_workflow.py` script:

```bash
python run_workflow.py
```

### 4. Closing the Environment

To manually deactivate the virtual environment, run the `close_env.sh` script:

```bash
./close_env.sh
```

## Usage

1. Navigate to the project directory.
2. Check the file `requirements.txt`, and run the setup script to initialize the environment if necessary.
3. Launch the notebook (`paper_figures.ipynb`) manually or using the provided Python script (`run_workflow.py`).
4. Run the Setup section to import required packages and set the matplotlib plotting style.
5. Use the numbered cell headings for each Figure.
6. Deactivate the virtual environment when done using the closing script.


## Using VS Code

- **Open the Project in VS Code:** Open the project directory in Visual Studio Code.
- **Select Python Interpreter:** Use the Command Palette (`Ctrl+Shift+P`) and select "Python: Select Interpreter" to choose the virtual environment.
- **Run Jupyter Notebooks:** Use the integrated Jupyter support within VS Code.
- **Automate Workflow:** Use VS Code tasks configured in `.vscode/tasks.json` to run the entire workflow.

## Support
For questions, issues, or suggestions, please contact the corresponding author.

## Authors and acknowledgment
Thanks to all authors who contributed to this project.

## License
This project is licensed under the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).

## Project status
This project is complete and primarily serves to store and provide access to the code and data used in the related publication. While the main development phase has concluded, minor revisions may still occur.
