# Midterm-ITAI-1371-Group-4
## TOPIC & URL

**Topic:** Predict whether a person will be making more than $50k 
**Dataset URL:** https://www.kaggle.com/datasets/emanfatima2025/adult-census-income-dataset 

## Team member contributions & notes

### Omar

### Maria Jose

### Janice Underwood

### Kim Nguyen
-	Dataset might contain some missing values and duplicate values, so when we clean the data, we need to identify which position is missing to fill in the NaN and Null. The duplicate ones are dropped. 
-	Using IQR to classify the unusual figure to determine what components will affect the annual salary of people. 
-	In this dataset, numerical and categorical are 2 types of variables, we need to convert categorical variable to numerical so that machine learning can understand and process them. With the workclass, it has order so using ordinal coding to assign number to this variable based on rank, while with marital status, it is easier to categorize it into 2 main parts, so one-hot coding is applied, etc. This step helps to make the dataset has a cleaner look.
-	Standardize the features to centered data around 0 mean and scaled it to unit variance, after that plotting bar graphs to compare before and after scaling. Min-Max scaling is used to normalize variance into [0, 1] range.
-	Based on the correlation heat map, we can see that: higher education levels slightly relate to longer weekly working hours, age has minimal linear impact on education or hours worked, each feature may contribute separately to predicting income.
- For machine learning:
- Problem Type:

Supervised classification: predict whether a person earns > 50K or ≤ 50K per year.

Feature Preparation

Numerical variables: age, education.num, hours.per.week, etc.

Categorical variables: workclass, marital.status, occupation, race, sex, etc.
→ Encode using One-Hot Encoding.

Handle missing values and detect outliers (using the IQR method).

Apply scaling/normalization (e.g., StandardScaler or MinMaxScaler).

- Model Candidates

Logistic Regression (for interpretability)

Decision Tree / Random Forest (for accuracy and feature importance)

Gradient Boosting (e.g., XGBoost, LightGBM)

Neural Network (for nonlinear relationships)

- Evaluation Metrics

Accuracy

Precision, Recall, and F1-Score

ROC-AUC for overall discriminative ability
