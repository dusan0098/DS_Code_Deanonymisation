# Code De-anonymisation using stylometry

A project for "Data Security" WS22/23 by Dusan Urosevic and Stefan Solarski.

## Contents:

The main file is a Jupyter **notebook** "Code De-anonymization notebook.ipynb". It already contains all of our outputs and visualizations, so it is not necessary to re-run the code.

The models, feature extraction method, and helper functions are stored in the **features** folder.

The sampled datasets, their features and the classification results are all pickled and stored in the **saved** folder.

## Setup:

Necessary packages are given in "requirements.txt".

The notebook can be run as. (we used Python v3.8.8).

Please download the whole github repo and run the notebook.

## Optional:

The original datasets, from kaggle, can be found at the following link: <https://www.kaggle.com/datasets/jur1cek/gcj-dataset>

If you wish to run the commented cells of the notebook that sample the datasets, the .cvs for 2009 and 2010 have to be put in a folder named data.

example location of file: data/gcj2009.csv/gcj2009.csv

## Acknowledgment

This work was based ideas presented in the paper "[De-anonymizing Programmersvia Code Stylometry](https://www.usenix.org/system/files/conference/usenixsecurity15/sec15-paper-caliskan-islam.pdf)"

Feature extraction was based on the code by 
## License

[MIT](https://choosealicense.com/licenses/mit/)
