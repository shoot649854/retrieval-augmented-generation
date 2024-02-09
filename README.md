[![Model architecture](https://img.shields.io/badge/Model%20Arch-Transformer%20Decoder-green)](#model-architecture)|
[![Model size](https://img.shields.io/badge/Params-5B-green)](#model-architecture)|
(#datasets)![AUR license](https://img.shields.io/badge/license-Apache%202-blue)


# retrieval-augmented-generation

To create the requirements.txt file for your Python project, run the following command using Poetry:

```bash
poetry export --format requirements.txt --output requirements.txt
pip3 freeze > requirements.txt
ln -s "/Volumes/volume/huggingface" "/Users/user/.cache"
```

To create the myenv.yaml file for your Conda environment, run the following command:

```bash
conda env export > myenv.yaml
```

## Installation
1. Install Miniconda:
Go to Miniconda and follow the installation instructions for Miniconda.
https://docs.anaconda.com/free/miniconda/

2. Check Existing Environments:
Make sure there is no existing environment named environment by running the following command:
```bash
conda info -e
```

3. Create Conda Environment:
Create a new Conda environment using the environment.yaml file:

```bash
conda env create -f environment.yaml
```

## Usage
1. Activate Environment:
Close the terminal application, then reopen it and activate the environment Conda environment:


```bash
conda activate environment
```

2. Select Kernel:
Make sure to select the environment python3.11 kernel in your Jupyter Notebook or JupyterLab environment.

3. Deactivate Environment:
Once you're done working, deactivate the Conda environment:


```bash
conda deactivate
```

This README.md provides instructions for setting up your project environment using Poetry and Conda. If you have any questions or encounter any issues, feel free to ask for assistance!
