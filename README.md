# Midterm-ITAI-1371-Group-4
## TOPIC & URL

**Topic:** Predict whether a person will be making more than $50k 
**Dataset URL:** https://www.kaggle.com/datasets/emanfatima2025/adult-census-income-dataset 

## Team member contributions & notes

### Omar

### Maria Jose

### Janice Underwood
Prediction whether a person will be making more than $50k

Library used: 
matplotlib.pyplot, numpy , pandas, sklearn.preprocessing (MinMaxScaler, OneHotEncoder, LabelEncoder, SimpleImputer and StandardScaler)

1.Data Loading & Exploration :  This is like opening up the data file and looking through it for the first time. check how many rows and columns, what kind of information is in each column (numbers, words, etc.)

**Technique used**: df.head() / df.shape / df.info() / df.isnull() / df.isnull().sum()

**2. Removed Unnecessary and Constant Columns**: 
Get rid of unnecessary columns to keep things clean. Separate Target variable from the dataset. Unnecessary columns can include:
Detect and remove such columns automatically:Check constant columns (only one unique value), Check quasi-identifiers (e.g., 'fnlwgt' in Adult dataset), 'fnlwgt' is a census weight — not useful for prediction.
**Technique used**:
df.drop, df[col].nunique() == 1,
Check constant columns (only one unique value): constant_cols = [col for col in df.columns if df[col].nunique() == 1]
Drop the unnecessary columns: df.drop(columns=unnecessary_cols, inplace=True, errors='ignore')

**3. Feature Engineering (Outliers, Binning, Domain Features)**
Manipulating the data to make it more useful for the model.
•	Outliers: look for extreme data points and decide whether to fix them or remove them because they can confuse the model.
•	Binning: This is like grouping numbers into ranges. For example, turning a person's exact age (25, 32, 48) into age groups (Young, Middle-aged, Senior). 
•	Domain Features: Using real-world knowledge (the "domain") to create new, helpful columns. E.g., Height and Weight, you can create a new feature called BMI.
 **Technique used** 
 pd.cut for binning, IQR(Interquartile Range) to remove outlier 
X['age_bin'] = pd.cut(X['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
Uses pd.cut() to divide ages into specified ranges and assigns readable labels like '<25', '25-35', etc., for easier analysis or modeling
                    
IQR(Interquartile Range):Function remove_outliers_iqr detects and caps outliers in specified numerical columns using the Interquartile                            Range (IQR) method. It limits values beyond the lower and upper bounds (Q1 − 1.5 × IQR, Q3 + 1.5 × IQR) using np.clip() and returns the                           cleaned DataFrame.
                    
**4. Filling NaN and Null Values (Handled Missing Values)**
Some entries are often blank. These "Not a Number" (NaN) or missing values will break the model.

**Technique used** 
Categorical: Fill with mode (most frequent) and Numerical: Fill with median and SimpleImputer
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='median')

**5. One-Hot Encoding (Encoded Categorical Features)**
OneHot Encoder converting categorical txt to numbers. I understood that One-Hot Encoding is how we turn these categories into numbers the model can understand, creating a clean DataFrame.

**Technique used**
sklearn.preprocessing.OneHotEncoder
fit: It learns the unique categories present in each column of X[categorical_cols].
transform: It applies the one-hot encoding logic to those columns, resulting in a NumPy array of 0s and 1s (because sparse_output=False).
encoder.get_feature_names_out(categorical_cols): After fitting, this method generates the new column names for the one-hot encoded data. The names follow a pattern like original_col_category_value

**6. Scaling (Standardization and Normalization): Adjusting the shape of the data's distribution
This step is all about making sure all the numbers are playing fair. If one column is Age (0 to 100) and another is Yearly Income (0 to 1,000,000), the income column will totally dominate the model.
 **Technique used**
• Normalization: Squishes all the numbers into a small, consistent range (like 0 to 1) so no single feature overpowers the Normalization: Adjust shape (MinMaxScaler)
MinMax Normalized version
•	Standardization: Adjusts the data's shape so that it has an average of 0 and a consistent spread, making it easier for some models to learn.
StandardScaler
 
**7. Correlation-Based Feature Reduction**
Correlation measures how closely two variables are related. If Years of Education and Degree Level are almost identical in what they tell you, you only need to keep one. 
**Technique used**
I use this analysis and a visualization (like a heat map) to cut out highly related, redundant features to speed up the model and keep it simple.

**Model Training & Evaluation**

**8. Train-Test Split**
**Technique used**
Before training the model, I split the data into two groups:
1.	Train Set: The large part the model studies and learns from.
2.	Test Set: The smaller part the model has never seen, for later to see how well the model truly performs on new data.

**09. Random Forest Model Training**
The Random Forest is the predictor tool. It's an algorithm that builds many individual "decision trees" (like flowcharts) and has them all vote on the final answer. 
**Technique used**
Train Set to "feed" this forest until it learns the patterns needed to make accurate predictions.

**10. Performance Evaluation and ROC**
Project: Predict whether a person will be making more than $50k
 
The code provides a solution to predict income bracket for making more than 50k using the adult.csv dataset,
covering all necessary steps from data cleaning and preprocessing to model training and evaluation using Logistic Regression.
Accuracy: The overall percentage of predictions the model got
Fi-Score: providing a balanced measure of performance.
AUC means evaluation. The model has a high ability to distinguish between the two income classes.

After training, I use test set to see how good the model is.
•	Evaluation: You check the model's accuracy (how often it's right) and other scores.
•	ROC (Receiver Operating Characteristic): This is a special way to measure the model's ability to correctly separate the two groups (making over $50k vs. making less). A perfect score is 1.0.

Result: Logistic Regression Model Evaluation
 Accuracy: 0.7807
F1-Score: 0.6477
AUC: 0.8822

Analysis & Final Goal
**11. Income Distribution – Target Variable Analysis**
**Technique used**
Before starting, it is a good idea to look closely at the Target Variable (the income groups). Check what percentage of people make over $50k versus under $50k. This helps to understand how balanced the prediction problem is and gives a baseline to beat.
Income Distribution Summary:
Count	Percentage
income		
<=50K	24720	75.92

>50K	7841	24.08

**12. Prediction: Making More Than $50k income Distribution** 
Target Variable Analysis on Test Data
**Technique used**
This is the final goal: Using the Random Forest model built, so I can now input a new person's details (age, education, job, etc.) and have the model predict whether that person will likely be making more than $50,000 or not.
 Income Distribution Summary (Test Set):
Count	Percentage
0	7417	75.92 

1	2352	24.08

### Kim Nguyen
