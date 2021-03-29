# Toxic Comment Classification Project
This repository contains codes developed as part of the Capstone Project requirement in Udacity's MLE Nanodegree programme
* Codes are developed based on Python on Windows 10 envrionment.
* Main libraries used for modelling are sklearn and tensorflow-keras.
* Main libraries used for preprocessing are nltk and sklearn.

# Code
##  Structure
The directory has the following file-structure (generate with `tree /f`)

```
toxic-comment
│   environment.yml
│   README.md
│
├───data
│       test.csv
│       test_labels.csv
│       train.csv
│
├───glove
│   │
│   ├───glove.6B
│   │       glove.6B.100d.txt
│   │
│   └───glove.twitter.27B
│           glove.twitter.27B.100d.txt
│
├───notebooks
        01-EDA.ipynb
        02a-Modelling_LR_Baseline.ipynb
        02b-Modelling_NbSVM.ipynb
        02c-Modelling_NN.ipynb
        common.py
```

## Set-up
Install conda environment
```
conda env create -f environment.yml
conda activate toxicc-env
```

## Usage
Go to to notebooks folder to run the scripts
* Note that data csv and glove vector txt files are tracked using Git LFS
* Need to setup Git LFS to be able to run the scripts