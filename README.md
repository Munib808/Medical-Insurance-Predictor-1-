# Medical Insurance Cost Predictor

This project predicts medical insurance charges based on personal attributes such as age, BMI, number of children, gender, smoking status, and region. It uses a machine learning pipeline that preprocesses the input data, applies feature engineering, and leverages an ensemble stacking regression model for robust predictions.

## Project Overview

Health insurance costs can vary significantly based on lifestyle and personal characteristics. This tool helps individuals estimate their insurance charges using a trained model based on historical data.

The project includes:
- Feature encoding and scaling
- A trained stacking regression model
- A web-based interface built with Streamlit for easy usage

## Dataset Information

The dataset includes the following features:

| Feature         | Description                                      |
|----------------|--------------------------------------------------|
| `age`          | Age of the primary beneficiary                   |
| `sex`          | Gender of the beneficiary                        |
| `bmi`          | Body Mass Index                                  |
| `children`     | Number of dependents covered by insurance        |
| `smoker`       | Smoking status (`yes` or `no`)                   |
| `region`       | Residential area in the US (`southeast`, etc.)   |
| `charges`      | Medical insurance cost (target variable)         |

## Model Architecture

We use a **Stacking Regressor**, which combines multiple regression models to enhance predictive performance.

### Base Models:
- **Random Forest Regression**
- **Decision Tree Regressor** .
- **Extra Trees Regressor**
- **XGB Regressor**

### Final Estimator:
- **Gradient Boosting Regressor**

### Preprocessing Pipeline:
- **StandardScaler** – Used for scaling numerical features.
- **pd.get_dummies()** – Used for encoding categorical variables.

All models and preprocessing steps were evaluated using cross-validation and combined into a final Stacking Regressor.

## Web Application

The project includes a **Streamlit-based web application** that allows users to:

- Input their data (age, BMI, etc.)
- Get a prediction of their insurance cost in real time
