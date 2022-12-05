

## Soybean annual yield in the United States 

#### Motivation

- Powered by machine learning approach, this work uses monthly climate condition to predict local annual soybean yield in the United States. 
- The overarching goal is to train a model using data from 1981 to 2015, and test the model's predictive power for annual yield in 2016.

#### Data source 

- GDHY (Global dataset of historical yields) provides annual crop yields: [lizumi and Sakai, 2020, The global dataset of historical yields for major crops 1981-2016](https://www.nature.com/articles/s41597-020-0433-7)

- [NLDAS](https://ldas.gsfc.nasa.gov/nldas) (The North American Land Data Assimilation System) provides monthly climate variables

#### Repository structure

- `./1.0_Data_ingestion.ipynb` - data ingestion and combination of yield and climate data
- `./2.0_EDA_feature_engineering_selection_OutputsCleared.ipynb` - exploratory data analysis and feature selection & enginnering based on domain knowledge
- `./3.0_Feature_selection_model_training.ipynb` - further feature selection based on sklearn methods, and model training
- `utils/` - utility functions
  - `__init__.py`
  - `preprocessing_EDA_utils.py` - for 1.0 and 2.0 notebooks
  - `feature_selection_model_training_utils.py` - for 3.0 notebook
- `data/`
  - `US_SoybeanYield_ClimateFeature_1981-2016_v2.csv` - data file containing final features and yield
- `webapp/` - deployment
  - `Dockerfile`
  - `server.py`
  - `model/` - xgboost best parameters and model
  - `img/` - figures saved from the notebooks
  - `templates/` - webpage content
  - `static/` - .css and .js scripts for style and web application

#### Demo

The trained XGBoost model was deployed on AWS EC2. See [here](http://www.ussoybean-demo.com/) for a demo website. 

#### License

This work is licensed under the [MIT License](https://github.com/wenwenkong/data-science-portfolio/blob/main/LICENSE).
