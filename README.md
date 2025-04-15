<h1>Project Overview</h1>

This project benchmarks multiple machine learning models to compare their performance in terms of accuracy and efficiency. The goal is to evaluate:

RMSE (Root Mean Squared Error): Measures the average magnitude of the error between predicted and actual values.

Training Time: The amount of time the model takes to train on the dataset.

Prediction Time: The amount of time the model takes to make predictions on the test set.

We use PySpark for distributed processing and large-scale data handling, and scikit-learn for traditional machine learning models. After training, the models are evaluated, and the performance metrics are visualized using bar plots for easy comparison.

<h1>Technologies Used</h1>

PySpark: Distributed computing framework to handle large-scale data and apply machine learning models.

scikit-learn: A machine learning library in Python used for various algorithms like Linear Regression, Random Forest, and MLP.

XGBoost: A powerful gradient-boosted decision tree library for efficient model training.

Matplotlib: A Python library used for data visualization, including the creation of bar plots for model comparison.

NumPy: A core library for numerical computing used to handle arrays and matrix operations.

<h1>Models Benchmarking</h1>

The following models are benchmarked in this project:

Linear Regression (LR): A simple linear model used to predict a continuous target variable.

Random Forest (RF): An ensemble method that creates multiple decision trees and averages their predictions.

Decision Tree (DT): A model that splits data into branches to make decisions based on feature values.

Gradient Boosted Trees (GBT): An ensemble method that builds models sequentially to correct the errors of previous models.

Support Vector Machine (SVM): A classifier that separates classes using hyperplanes in high-dimensional space.

XGBoost: An optimized version of gradient boosting with better performance.

K-Nearest Neighbors (KNN): A simple, instance-based learning algorithm that makes predictions based on the closest neighbors.

Multi-layer Perceptron (MLP): A deep learning model with one or more hidden layers for complex prediction tasks.

<h1>How the Models are Evaluated</h1>

Each model is evaluated based on the following metrics:

RMSE (Root Mean Squared Error): This is computed using the predictions made by the model on the test set. It gives a measure of how well the model is performing, with lower values indicating better accuracy.

Training Time: The time taken by the model to train on the training dataset.

Prediction Time: The time taken by the model to generate predictions on the test set.

The models are evaluated using a combination of PySpark's RegressionEvaluator for distributed models and scikit-learn's mean_squared_error for traditional models like MLP and KNN.
