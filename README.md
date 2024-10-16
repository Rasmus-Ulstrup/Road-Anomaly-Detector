# Road_Anomaly_Detector

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Road Anomaly Detector** is a machine learning project aimed at identifying and detecting anomalies on roads using image processing techniques. This project is designed to assist in the automation of road maintenance by providing a reliable tool for identifying potholes, cracks, and other surface irregularities.

## Features

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rasmus-Ulstrup/Road-Anomaly-Detector.git
   cd Road-Anomaly-Detector
2. **Create a conda environment**:
    ```bash
    conda env create -f environment.yml
3. **Activate the enviroment**:
    ```bash
    conda activate road_anomaly_detector
4. **Download and install pylon**:
   ```bash
   https://www2.baslerweb.com/en/downloads/software-downloads/software-pylon-7-5-0-windows/

## License
This project is licensed under the MIT License.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         road_anomaly_detector and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── road_anomaly_detector   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes road_anomaly_detector a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

