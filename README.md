# Midterm-ITAI-1371-Group-4
# IMPORTANT, PLEASE READ PROF. VISHWA
### Repo structure
- Our notes and *individual reports* can be found in the Readme file, separated by name of each student.
- **Final code submission is *FINAL GROUP 4 MIDTERM_DATA_SET.ipynb***
- All additional *.ipynb* are separate individual contributions. 

## TOPIC & URL

**Topic:** Predict whether a person will be making more than $50k 
**Dataset URL:** https://www.kaggle.com/datasets/emanfatima2025/adult-census-income-dataset 

## Team member contributions & notes

### Omar
**STEPS TAKEN UP TO DATA VALIDATION**
- Loaded Data: imported the dataset and performed initial checks on its data types and basic information.Initial Cleaning: I transformed all question mark ("?") characters into null values to correctly represent the missing data.
- Data Quality Checks: checked for missing values, duplicate rows, and performed an initial outlier check using the IQR method.
- Statistical Analysis: calculated basic summary statistics (mean, median, mode, etc.) for my numerical variables and complementary statistics (skewness, kurtosis) to understand my data's shape. I also generated a separate summary for all categorical variables.
- Variable Removal: evaluated and removed four specific columns using the .drop() method:fnlwgt, education (the string version, as I kept the integer version for statistcal analysis), capital.gain, and capital.loss.
- Data Visualization: used three types of plots to inspect my data:Histograms to see the distribution of my numerical variables. Box plots to visually identify outliers. Bar plots/Pie charts to check the frequencies of my categorical variables.
- Correlation Analysis: created a correlation matrix and scatter plots to understand the relationships between my numerical variables.
- Validation Planning: planned to perform a "before and after" comparison of the correlation matrices to validate my entire cleaning process.

**JUSTIFICATION OF REMOVED VARIABLES**
- The variable *fnglwgt* is undefined and no documentation was found from the source to explain what this variable does. It is assumed that it is a calculated value from the previous data handler.
- The *education* variable appears to be duplicated in the dataset, but one is a string and the other is integer. We kept the integer version since we can do our statsistical analysis with it instead of having to convert the string values.
- The *capital.gain* and *capital.loss* variables both have the value of 0 approx 96% and 98%, respectively, across all rows. 

### Maria Jose

### Janice Underwood
Prediction whether a person will be making more than $50k

Library used: 
matplotlib.pyplot, numpy , pandas, 
sklearn.preprocessing import StandardScaler, 
sklearn.preprocessing import MinMaxScaler
sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

1.Data Loading & Exploration :  This is like opening up the data file and looking through it for the first time. check how many rows and columns, what kind of information is in each column (numbers, words, etc.)
Code: df / df.shape / df.info() / df.isnull() / df.isnull().sum()

2. Removed Unnecessary and Constant Columns: 
Get rid of unnecessary columns to keep things clean. Separate Target variable from the dataset. Unnecessary columns can include:
•	Identifier-like columns (e.g., 'id', 'fnlwgt' if not useful)
•	Duplicate or constant-value columns
•	Columns with too many unique values relative to dataset size
•	Irrelevant for modeling (like names or timestamps)
Detect and remove such columns automatically.
•	Check constant columns (only one unique value)
•	Check quasi-identifiers (e.g., 'fnlwgt' in Adult dataset)
•	'fnlwgt' is a census weight — not useful for prediction.
•	Drop the unnecessary columns
Separate Target Variable (predict income )

3. Feature Engineering (Outliers, Binning, Domain Features)
Manipulating the data to make it more useful for the model.
•	Outliers: look for extreme data points and decide whether to fix them or remove them because they can confuse the model.
•	Binning: This is like grouping numbers into ranges. For example, turning a person's exact age (25, 32, 48) into age groups (Young, Middle-aged, Senior). 
•	Domain Features: Using real-world knowledge (the "domain") to create new, helpful columns. E.g., Height and Weight, you can create a new feature called BMI.

4. Filling NaN and Null Values (Handled Missing Values)
some entries are often blank. These "Not a Number" (NaN) or missing values will break the model, so you have to carefully fill them in
Strategy: Categorical: Fill with mode (most frequent) and Numerical: Fill with median

5. One-Hot Encoding (Encoded Categorical Features)
OneHot Encoder converting categorical txt to numbers. Computers like numbers, but your data might have words , I understood that One-Hot Encoding is how we turn these categories into numbers the model can understand, creating a clean DataFrame. Generating meaningful column names and storing the result in a new, manageable DataFrame for easy merging or concatenation with other features.

I used this method for two things:
•	fit: It learns the unique categories present in each column of X[categorical_cols].
•	transform: It applies the one-hot encoding logic to those columns, resulting in a NumPy array of 0s and 1s (because sparse_output=False).
•	encoder.get_feature_names_out(categorical_cols): After fitting, this method generates the new column names for the one-hot encoded data. The names follow a pattern like original_col_category_value
 
One-hot encode categorical columns
Result:
encoder = OneHotEncoder(drop='first', sparse_output=False)

encoded_cols = pd.DataFrame(
    encoder.fit_transform(X[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols)

6. Scaling and Standardizing: Changing the range the data - Scaled and normalized data. Scaling Normalization: Adjusting the shape of the data's distribution
This step is all about making sure all the numbers are playing fair. If one column is Age (0 to 100) and another is Yearly Income (0 to 1,000,000), the income column will totally dominate the model.
•	Scaling/Normalization: Squishes all the numbers into a small, consistent range (like 0 to 1) so no single feature overpowers the oNormalization: Adjust shape (MinMaxScaler)
MinMax Normalized version

•	Standardization: Adjusts the data's shape so that it has an average of 0 and a consistent spread, making it easier for some models to learn.

7. Correlation-Based Feature Reduction
Correlation measures how closely two variables are related. If Years of Education and Degree Level are almost identical in what they tell you, you only need to keep one. You use this analysis and a visualization (like a heat map) to cut out highly related, redundant features to speed up the model and keep it simple.

Model Training & Evaluation
8. Train-Test Split
Before training the model, you split your data into two groups:
1.	Train Set: The large part the model studies and learns from.
2.	Test Set: The smaller part the model has never seen, for later to see how well the model truly performs on new data.

09. Random Forest Model Training
The Random Forest is the predictor tool. It's an algorithm that builds many individual "decision trees" (like flowcharts) and has them all vote on the final answer. We may Train Set to "feed" this forest until it learns the patterns needed to make accurate predictions.

10. Performance Evaluation and ROC
Project: Predict whether a person will be making more than $50k
 
The code provides a solution to predict income bracket for making more than 50k using the adult.csv dataset,
covering all necessary steps from data cleaning and preprocessing to model training and evaluation using Logistic Regression.
Accuracy: The overall percentage of predictions the model got
Fi-Score: providing a balanced measure of performance.
AUC means evaluation. The model has a high ability to distinguish between the two income classes.

After training, I use test set to see how good the model is.
•	Evaluation: You check the model's accuracy (how often it's right) and other scores.
•	ROC (Receiver Operating Characteristic): This is a special way to measure the model's ability to correctly separate the two groups (making over $50k vs. making less). A perfect score is 1.0.

Result:
--- Logistic Regression Model Evaluation ---
 Accuracy: 0.7807
F1-Score: 0.6477
AUC: 0.8822

Analysis & Final Goal
11. Income Distribution – Target Variable Analysis
Before starting, it is a good idea to look closely at the Target Variable (the income groups). Check what percentage of people make over $50k versus under $50k. This helps to understand how balanced the prediction problem is and gives a baseline to beat.
 
 Income Distribution Summary:
Count	Percentage
income		
<=50K	24720	75.92

>50K	7841	24.08

12. Prediction: Making More Than $50k income Distribution 
Target Variable Analysis on Test Data
•	This is the final goal: Using the Random Forest model built, so I can now input a new person's details (age, education, job, etc.) and have the model predict whether that person will likely be making more than $50,000 or not.
 Income Distribution Summary (Test Set):
Count	Percentage
0	7417	75.92
    
1	2352	24.08

Works Cited
Google page: research, writing and spelling correction 
Google Collab

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

Neural Network (for nonlinear relationships)

- Evaluation Metrics

Accuracy

Precision, Recall, and F1-Score

ROC-AUC for overall discriminative ability
