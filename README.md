# PROJECT TITLE: Heart-Disease-detection

## Overview of the Project:

This project aims to build a machine-learning model that predicts whether a patient has heart disease based on various medical attributes. 
The dataset includes clinical parameters such as age, chest pain type, cholesterol level, resting blood pressure, maximum heart rate achieved, and more.
Using Exploratory Data Analysis (EDA), visualizations, and a K-Nearest Neighbors classification model, 
the project identifies the patterns and correlations within the features and evaluates the model's prediction accuracy

## Features:
1. Data Loading & Exploration
– Displaying dataset structure (head(), sample(), info(), describe())
– Checking class distribution using value_counts()

2. Data Cleaning
– Removing duplicate values
– Splitting data into features and target variable

3. Data Visualization
– Bar graph showing number of patients with/without heart disease
– Correlation heatmap using Seaborn

4. Data Preprocessing
– Feature scaling using StandardScaler()
– Train-test split (80–20 ratio)

5. Model Building
– Training a K-Nearest Neighbors (KNN) classifier
– Predicting test output
– Predicting heart disease for a new example input

6. Model Evaluation
– Calculating accuracy using accuracy_score()

## Technologies / Tools Used:

1.Programming Language: Python

2.Libraries:
  Data handling: pandas, numpy
  Visualization: matplotlib, seaborn
  Machine Learning: scikit-learn (StandardScaler, train_test_split, KNeighborsClassifier, accuracy_score)
 
3.Environment: Anaconda Navigator / Jupyter Notebook

## Steps to Install & Run the Project:
1. Install Anaconda (recommended)
Download and install Anaconda:
https://www.anaconda.com/products/distribution

2. Create and Activate a Virtual Environment (optional)
conda create -n heart_project python=3.10
conda activate heart_project
3. Install Required Libraries
Most come pre-installed in Anaconda, but you can install manually if needed:
                                                pip install numpy pandas matplotlib seaborn scikit-learn
4. Launch Jupyter Notebook
          jupyter notebook

6. Open Your Project Notebook
      Place heart_disease_data.csv in the same folder as your notebook.
      Run the cell:
          df = pd.read_csv('heart_disease_data.csv')
   
7. Execute the Notebook Cells
Run each cell in order:
Importing libraries
Loading data
EDA and visualizations
Preprocessing
Model training
Prediction

## Instructions for Testing the Model:
1. Evaluate the Model Accuracy
After training KNN:

accuracy_score(y_test, y_pred)
This gives the model’s performance on unseen data.

2. Test with a New Patient Record
Example:

ex_1 = (37,1,3,145,233,1,0,150,0,2.3,0,0,1)
new = np.array(ex_1)
knn.predict(new.reshape(1,-1))
Output:

0 → No Heart Disease

1 → Heart Disease Present

3. Modify Input for Different Test Cases
You may change the tuple values to simulate different medical conditions.

## Screenshots:
<img width="1093" height="709" alt="Screenshot 2025-11-23 184704" src="https://github.com/user-attachments/assets/cb03c81b-6553-4404-a5ee-95317cb14a7e" />
<img width="1488" height="490" alt="Screenshot 2025-11-23 184748" src="https://github.com/user-attachments/assets/7ca12a00-443f-459e-a00c-b953d2ea1f0c" />
<img width="1476" height="433" alt="Screenshot 2025-11-23 184832" src="https://github.com/user-attachments/assets/db383544-2798-40d7-9fd5-170b8a95bb4c" />
<img width="1480" height="719" alt="Screenshot 2025-11-23 184902" src="https://github.com/user-attachments/assets/e0deb7b9-2a37-4f81-9a4b-26c7f88b7d17" />
<img width="1507" height="745" alt="Screenshot 2025-11-23 184923" src="https://github.com/user-attachments/assets/b2bb1a14-5f6d-4625-8054-274aa694fd40" />
<img width="1480" height="776" alt="Screenshot 2025-11-23 184950" src="https://github.com/user-attachments/assets/415c078b-2524-4639-9623-dc3a6d5baba6" />
<img width="1471" height="683" alt="Screenshot 2025-11-23 185013" src="https://github.com/user-attachments/assets/54a86583-afe1-40b8-a399-a7bacb2dbb70" />
<img width="1514" height="756" alt="Screenshot 2025-11-23 185059" src="https://github.com/user-attachments/assets/186a6806-c79e-4cbb-b110-0371094919e2" />
<img width="1418" height="347" alt="Screenshot 2025-11-23 185118" src="https://github.com/user-attachments/assets/89e4a1e0-8bbc-4ef4-bcb5-98ef3622e9c3" />


