
# Job Market Salary Prediction using PySpark

## Abstract

In the era of big data, accurate job market analysis has become essential for understanding employment trends, forecasting salaries, and aiding job seekers and organizations in informed decision-making. This project focuses on building a predictive model to estimate job salaries based on job titles and locations using various machine learning algorithms implemented in **PySpark**, a powerful framework for large-scale data processing.

We benchmarked a range of machine learning models including:
- Linear Regression (LR)
- Random Forest (RF)
- Decision Tree (DT)
- Gradient Boosted Trees (GBT)
- Support Vector Machine (SVM)
- XGBoost
- K-Nearest Neighbors (KNN)
- Multi-layer Perceptron (MLP)

Models were evaluated based on:
- Training time
- Prediction time
- Root Mean Square Error (RMSE)

Among all models, **XGBoost** emerged as the top performer, achieving the lowest RMSE with fast training and prediction times.

---

## Keywords

Job Market Analysis, Salary Prediction, Machine Learning, Model Benchmarking, PySpark, XGBoost

---

## Objective

- Analyze the job market focusing on salary prediction based on job title and job location.
- Benchmark multiple machine learning models to determine the most efficient and accurate one.
- Evaluate models using training time, prediction time, and RMSE.
- Identify the best model for deployment.
- Fully implement the system in **PySpark** for scalability.
- Predict salaries for unseen job titles and locations, offering insights for job seekers, HR professionals, and policymakers.

---

## Proposed System

### 1. Data Scraping
- Scraped job listings from platforms like Glassdoor and Indeed.
- Extracted fields: job title, job location, salary estimates, company name, and job description.
- Used Python tools like **BeautifulSoup** and **Selenium**.

### 2. Data Preprocessing
- Cleaned and structured the data using **PySpark**.
- Handled missing values, duplicates, and irrelevant data.
- Encoded categorical features and split the dataset into training and testing sets.

### 3. Model Benchmarking
Implemented and evaluated:
- Linear Regression (LR)
- Random Forest (RF)
- Decision Tree (DT)
- Gradient Boosted Trees (GBT)
- Support Vector Machine (SVM)
- XGBoost
- K-Nearest Neighbors (KNN)
- Multi-layer Perceptron (MLP)

Evaluation Metrics:
- **Training Time**
- **Prediction Time**
- **Root Mean Square Error (RMSE)**

### 4. Model Selection and Deployment
- **XGBoost** was selected based on the best RMSE and computational efficiency.
- Deployed the XGBoost model for predicting unseen salary data.

### 5. Implementation Platform
- Complete implementation using **PySpark** to leverage distributed computing for handling large-scale datasets.

---

## Benchmarking Results

- **XGBoost** achieved the lowest RMSE of **24,586.38**.
- Training Time: **0.084 seconds**
- Prediction Time: **0.0018 seconds**

Other models like Linear Regression, SVM, Random Forest, Decision Tree, GBT, KNN, and MLP showed higher RMSE or longer training/prediction times, making XGBoost the most suitable.

---

## Technologies Used

- PySpark
- Machine Learning Algorithms (XGBoost, RF, DT, etc.)
- Python (BeautifulSoup, Selenium)
- Big Data Handling

---

## Future Enhancements

- Deploy model predictions as RESTful APIs for easy integration.
- Use real-time job market data feeds.
- Extend prediction features (e.g., based on required skills, company reputation).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributors

- [Your Name](https://github.com/your-username) - Developer

Open to collaborations and improvements. Feel free to contribute!

---
