# Stylometric approach to code de-anonymization

A project for "Data Security" WS22/23 by Dusan Urosevic and Stefan Solarski.

## Contents:

The main file is a Jupyter **notebook** "[Stylometric De-anonymization.ipynb](https://github.com/dusan0098/DS_Code_Deanonymisation/blob/main/Stylometric%20De-anonymization.ipynb)". It already contains all of our outputs and visualizations, so it is not necessary to re-run the code.

The feature extraction methods, classification models, and helper functions are stored in the **functions** folder.

The sampled datasets*, their features, and the classification results are all pickled and stored in the **saved** folder. 

*This had to be done due to the original .csv files being over the github size limit. 

## Setup:

Clone the repository and use [pip](https://pip.pypa.io/en/stable/), or another package manager, to install the requirements.

```bash
git clone https://github.com/dusan0098/DS_Code_Deanonymisation.git
cd DS_Code_Deanonymisation
pip install -r requirements.txt
```

Necessary packages are given in "requirements.txt".

The notebook can be run as is. (we used Python v3.8.8).

## Optional:

The original kaggle datasets can be found [here](https://www.kaggle.com/datasets/jur1cek/gcj-dataset). 

If you wish to run the commented cells of the notebook that sample the datasets, the .csv files for 2009 and 2010 have to be put in a folder named **data**.

example location of file: data/gcj2009.csv/gcj2009.csv

## Acknowledgment

This work was based on ideas presented in the paper "[De-anonymizing Programmers via Code Stylometry](https://www.usenix.org/system/files/conference/usenixsecurity15/sec15-paper-caliskan-islam.pdf)"

Functions for working with the abstract syntax tree (AST) using [javalang](https://github.com/c2nes/javalang) were based on code by [Yurii Rebryk](https://github.com/rebryk/code_stylometry).

## License

[MIT](https://choosealicense.com/licenses/mit/)
