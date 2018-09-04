# AIS - LSTM

About
------------
LSTM Keras Neural Network to predict ship location using Danish AIS data

Project Folder Structure
------------

    ├── README.md          <- The top-level README.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling. In .npz format.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── pull_danish_data.py
    │   │
    │   ├── preproc		   <- Scripts to turn raw data into features for modeling
    │   │   ├── danish_preproc.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── danish_predict_model.py
    │   │   ├── danish_train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       ├── danish_graph_tracks.py
    │
	│
	│
	└────────────────────────────────────────────────────────────────────────────────────────────────────
--------
