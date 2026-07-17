# Iris Flower Classification Using Machine Learning

## Project Overview

This Flask web application classifies an Iris flower as **Iris-setosa**, **Iris-versicolor**, or **Iris-virginica** from four measurements. It was created as Artificial Intelligence Internship Project 2.

## Internship Task 2 Objective

Build a complete machine learning workflow: load and analyze a dataset, train and test a supervised classifier, evaluate it, save it, and use it through a simple web interface.

## Features

- Loads the official UCI Iris dataset with Pandas.
- Prints basic analysis, including missing values and class distribution.
- Uses an 80/20 stratified train-test split with `random_state=42`.
- Fits `StandardScaler` only on training data to prevent data leakage.
- Tests KNN values from 1 to 15 and selects the best testing accuracy.
- Evaluates accuracy, weighted precision, recall, F1 score, report, and confusion matrix.
- Saves the trained KNN model, fitted scaler, and metrics with Joblib/JSON.
- Provides input validation and real predictions in Flask.

## Dataset Information

The Iris dataset has 150 samples, 4 numeric features, and 3 species. The `data/iris.data` file has no header, so the application assigns these columns: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, and `species`.

## Machine Learning Workflow

1. Load and clean the data.
2. Separate the four features and species target.
3. Split data into training (80%) and testing (20%) sets.
4. Fit a scaler on training features and transform both sets.
5. Compare KNN models for K values 1 through 15.
6. Train and evaluate the final model with the best K.
7. Save the model artifacts for Flask predictions.

## KNN Algorithm Explanation

K-Nearest Neighbors classifies a new flower by finding its nearest training examples. The flower receives the species most common among those neighbours. Feature scaling is used so measurements on different ranges contribute fairly to distance calculations.

## Project Structure

```text
iris-ai-classifier/
├── app.py
├── train_model.py
├── requirements.txt
├── data/iris.data
├── model/                  # Created/filled after training
├── static/css/style.css
├── static/js/script.js
└── templates/index.html
```

## Technologies Used

Python, Flask, Pandas, NumPy, scikit-learn, Joblib, HTML5, CSS3, and basic JavaScript.

## Installation Instructions

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell prevents activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## How to Train the Model

Make sure `data/iris.data` exists, then run:

```powershell
python train_model.py
```

This creates `model/knn_model.pkl`, `model/scaler.pkl`, and `model/metrics.json`.

## How to Run the Flask Application

```powershell
python app.py
```

Open the local URL displayed by Flask, normally `http://127.0.0.1:5000`.

## Model Evaluation

The exact metrics are printed after each training run and stored in `model/metrics.json`. The web page reads this file dynamically, so it always displays metrics from the saved model.

## How Prediction Works

The form collects features in this fixed order: sepal length, sepal width, petal length, petal width. Flask validates the values, transforms them with the saved scaler, and passes them to the saved KNN model.

## Future Improvements

- Add a confusion matrix visualization.
- Compare additional classification algorithms.
- Add automated tests and deployment configuration.
