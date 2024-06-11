# Geospatial Anomaly Detection

## Table of Contents
1. Introduction
2. File Structure
3. Setup Instructions
4. Usage
5. Contributing
6. License
7. Contact
8. Acknowledgments

## Introduction
Repository for the GeoAD framework. Contains different implementations of Autoencoders for anomaly detection in
geospatial data.
Content submited to the IEEE Sensors Conference 2024 as

```
Vitor Elias, John Dehls, Pierluigi Salvo Rossi, "On the Impact of Spatial and Temporal Modeling for Autoencoders in Geospatial Anomaly Detection"
```


## File Structure

```
.
├── dev
│   ├── Hyperparameter_tuning
│   └── test_models.py
├── geoad
│   ├── nn
│   └── utils
├── notebooks
│   ├── models
│   ├── anomaly_showcase.ipynb
│   ├── format_dataset.ipynb
│   └── result_analysis.ipynb
├── environment.yml
├── README.md
├── setup.py
└── LICENSE

```

- `dev`: Simulation files for experiments in Sensors 2024 paper
- `geoad`: Geopastial Anomaly Detection package.
    - `nn`: source codes for models
    - `utils`: implementation of utility functions
- `notebooks`: Notebooks for model showcasing
- `environment.yml`: Conda environment file
- `LICENSE`: License file for use and sharing of this material

## Setup Instructions
```
conda env create -f environment.yml
pip install -e .
```

## Usage
From geoads.nn.models, import the desired autoencoder. 


## InSAR Anomaly Detection
You can replicate the InSAR data anomaly detection experiment.  

    - Format_dataset.ipynb generates a dataframe using the csv from InSAR data
    - With the content from dev/hyperparameter_tuning, you obtain the optimal hyperparameters
    - With dev/test_models, you obtain the AUC scores for a test portion of the data.



## License
(Refer to the `LICENSE` file for the project's licensing information.)
