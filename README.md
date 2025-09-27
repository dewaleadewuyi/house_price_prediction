

# 🏡 Housing Price Prediction – Machine Learning Project

## 📌 Project Overview

This project is part of my Machine Learning Internship (Month 2) tasks. The goal is to build a **machine learning regression model** that predicts **housing prices in California** based on various features such as location, number of rooms, population, and more.
The project demonstrates a typical end-to-end machine learning workflow: **data preprocessing, feature selection, model training, evaluation, and result visualization.**

---

## 🧠 Project Objectives

* Load and explore the California Housing dataset.
* Clean and preprocess the data for training.
* Select key features that influence housing prices.
* Train a machine learning regression model.
* Evaluate the model’s performance using metrics like MSE, RMSE, and R² score.
* Visualize actual vs predicted prices for performance analysis.

---

## 🛠️ Technologies Used

* **Python 3.x** – Programming language
* **Pandas & NumPy** – Data manipulation and analysis
* **Matplotlib** – Data visualization
* **scikit-learn** – Machine learning modeling and evaluation

---

## 📊 Workflow

1. **Data Loading:** Import the California Housing dataset from `scikit-learn`.
2. **Exploration & Cleaning:** Inspect data shape, features, and handle any missing values.
3. **Feature Selection:** Choose important features like `MedInc`, `AveRooms`, `Population`, etc.
4. **Data Splitting:** Split the dataset into training and testing sets (80/20).
5. **Feature Scaling:** Standardize the feature values using `StandardScaler`.
6. **Model Training:** Train a **Linear Regression** model on the training set.
7. **Model Evaluation:** Use metrics such as **MSE**, **RMSE**, and **R² score** to assess performance.
8. **Visualization:** Plot **Actual vs Predicted** house prices for a visual performance check.

---

## 📈 Results

The trained regression model successfully predicts housing prices with good accuracy.
Sample performance metrics:

* **Mean Squared Error (MSE):** ~0.52
* **Root Mean Squared Error (RMSE):** ~0.72
* **R² Score:** ~0.60 – 0.70

The results demonstrate how machine learning can assist in real estate decision-making by estimating property values from available features.

---

## 🚀 Future Improvements

* Experiment with more powerful models like **Random Forest Regressor** or **Gradient Boosting** to improve accuracy.
* Add more feature engineering steps to enhance the dataset.
* Deploy the model as a web API using **Flask** or **FastAPI** for real-world usage.

---

## 📂 How to Run the Project

### 1. Clone this repository:

```bash
git clone https://github.com/your-username/HousingPricePrediction.git
```

### 2. Navigate into the project folder:

```bash
cd HousingPricePrediction
```

### 3. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
.\venv\Scripts\activate   # on Windows
source venv/bin/activate  # on Mac/Linux
```

### 4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 5. Run the script:

```bash
python housing_price_prediction.py
```



Would you like me to write a **matching README for Task 4 (Iris Classification)** too, so both repos look equally professional on GitHub?
