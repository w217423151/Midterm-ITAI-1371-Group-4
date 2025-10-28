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
ITAI 1371
Professor Viswanatha  Rao
October 28, 2025
Group 4. Maria Jose Viveros

Data Preprocessing and Cleaning for the Adult Income Dataset

The goal of this project is to predict which factors influence whether a person earns more or less than $50,000 per year. The dataset includes 32,561 participants and 15 variables representing demographic and socioeconomic characteristics. These variables are both numerical (e.g., age and hours worked per week) and categorical (e.g., education, occupation, and income level).
 
In the overall data quality analysis, the dataset initially contained 32,561 rows, with 24 duplicates accounting for just 0.07 percent of the data. After removing these duplicates, the final dataset had 32,537 unique rows. Missing values were found in the occupation (5.66 percent), workclass (5.63 percent), and native-country (1.79 percent) columns. Using the command df.replace('?', pd.NA, inplace=True) was crucial because it replaced question marks with recognized missing values, enabling proper handling of null data during cleaning and analysis. To fill in missing data, mode imputation was used for categorical variables with nulls. The occupation variable was filled with the mode “Prof-specialty,” workclass with “Private,” and native-country with “United-States.” Mode imputation was suitable here because it maintained the original structure of the categorical variables without changing the distribution of the dominant categories. Since the distribution after imputation remained nearly the same as initially, this approach helps ensure the model will not be biased by over-replacements. As a result, the dataset quality stays stable and ready for further analysis or modeling.
 
Certain variables were removed from the dataset to enhance data quality and model performance. The fnlwgt variable was excluded because it is undefined and likely acts as a combined indicator from a previous calculation. The string form of the education variable was removed since it duplicates the information found in education.num. Additionally, capital.gain and capital.loss were eliminated due to their extreme imbalance, with zeros in 96 and 98 percent of cases, respectively, resulting in highly skewed distributions that contribute little to predictive analysis. The correlations among numerical variables (e.g., education.num, hours-per-week, age, etc.) were generally low (below 0.5), with the exception of a strong correlation between education and education.num, since both represent the same data—one as a string and the other as a number. To avoid redundancy, one of these variables was removed.
 
Because categorical variables are stored as text strings, they need to be converted to numerical format for machine learning algorithms. Initially, One-Hot Encoding was tested, but it created too many new variables, which is effective for logistic or linear regression models but inefficient for decision tree algorithms. Therefore, label encoding was used for categorical variables, helping to produce a more balanced distribution, preserve semantic meaning, and reduce bias related to gender, occupation, and country of origin. For example, the marital.status variable was recoded because the original data showed a highly uneven distribution. “Married-civ-spouse” made up 46.01 percent of the sample, while other categories like “Married-AF-spouse” had only 0.07 percent, creating imbalance. To create a more uniform distribution and reduce sparsity, categories were grouped into two broader classes: “Married,” which includes Married-civ-spouse, Married-spouse-absent, and Married-AF-spouse, and “Not Married,” which includes Never-married, Divorced, Separated, and Widowed. This recoding resulted in a balanced split of 47.36 percent and 52.64 percent, simplifying the variable and improving model efficiency.

To prepare numerical features for modeling, both age and hours-per-week were scaled using two methods: Standard Scaling (Z-score) and Min-Max Scaling (0-1 normalization). Standard Scaling adjusted the features to have a mean of 0 and a standard deviation of 1, while Min-Max Scaling normalized them between 0 and 1. Histograms comparing original and scaled data showed that scaling preserved the data shape while adjusting the range.

Outliers in these variables were treated using the Interquartile Range (IQR) method. Values below Q1 - 1.5 IQR or above Q3 + 1.5 IQR were capped to these bounds. This approach reduced extreme values while maintaining the overall distribution. Boxplots before and after capping confirmed that the influence of outliers was minimized, ensuring more stable input for machine learning models.

In summary, the Adult Income dataset was carefully preprocessed for machine learning. Missing values in categorical variables were handled using mode imputation, ensuring no null data remained. Redundant and highly skewed columns such as fnlwgt, education (string), capital.gain, and capital.loss were removed to improve data quality. Categorical variables were encoded with label encoding, and categories with high imbalance, such as marital.status, were recoded for better distribution. Numerical features like age and hours-per-week were scaled with both Standard Scaling and Min-Max normalization, and outliers were capped using the IQR method to reduce their impact.



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
