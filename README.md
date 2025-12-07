README 
Heart Disease Prediction using Machine Learning

This project develops a machine learning classification system to predict the likelihood of heart disease based on patient data. It includes data preprocessing, model training, hyperparameter tuning, and evaluation across multiple algorithms.
 
Project Overview
This project uses a dataset of 1,025 patient records containing medical and demographic features.
The goal is to build a reliable binary classifier to identify:
•	0 → No Heart Disease
•	1 → Heart Disease
The project compares multiple machine learning models and evaluates their performance using several metrics.
 
Models Used
The following machine learning algorithms were trained and evaluated:
•	Logistic Regression
•	Decision Tree Classifier
•	Random Forest Classifier
•	Support Vector Machine (SVM)
Each model was tuned using GridSearchCV to improve performance.
 

Workflow Summary

1. Data Preparation
•	Verified dataset integrity
•	Encoded categorical variables
•	Scaled numerical features using StandardScaler
•	Applied a 70/15/15 train, validation, and test split

2. Model Development
Each model was trained and evaluated using:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	Confusion Matrix
•	ROC Curve






3. Best Performing Model
The Support Vector Machine (SVM) model achieved the most balanced performance with strong recall, making it the recommended final model for this task.
 
Key Results Summary
Model	Summary
Logistic Regression	Solid baseline performance
Decision Tree	Overfitting observed
Random Forest	Very high training scores indicating possible overfitting
SVM	Best overall model and best recall
 
Repository Structure
notebook.ipynb               - Full machine learning workflow  
ML Final Report Ali.docx    - Detailed written report  
README.md                  - Project documentation  
 
Technologies Used
•	Python
•	Pandas
•	NumPy
•	Scikit-learn
•	Matplotlib
•	Seaborn
•	Jupyter Notebook
 
How to Run
Install required libraries:
pip install -r requirements.txt
Open the notebook:
jupyter notebook notebook.ipynb
 
Future Improvements
•	Experiment with advanced models such as XGBoost or LightGBM
•	Add SHAP or LIME for model interpretability
•	Deploy the model using Streamlit or Flask
•	Expand dataset and perform more feature engineering
 
Author
Ali Oraiqat
Data Scientist

<img width="468" height="635" alt="image" src="https://github.com/user-attachments/assets/b000234d-955e-4a5a-a525-28e5c61ecdd3" />
