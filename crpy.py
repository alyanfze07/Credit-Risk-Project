# %% [markdown]
# Data Understanding

# %%
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn import preprocessing, compose, model_selection, ensemble, linear_model, neighbors, metrics

# %%
df = pd.read_excel("loan_data_2007_2014.xlsx")
print(df.head(5))

# %% [markdown]
# Data Understanding

# %%
df.info()

# %%
mv = df.isnull().sum()
mv_sorted = mv.sort_values()
mv_sorted

# %%
df.isnull().sum().sum()

# %%
#duplicate rows check
duplicate_rows = df[df.duplicated()]
number_of_duplicates = duplicate_rows.shape[0]
print(number_of_duplicates)

# %% [markdown]
# Feature Engineering

# %%
df_1 = df.drop(columns=['Unnamed: 0', 'id', 'member_id', 'funded_amnt_inv', 'installment', 'issue_d', 'url', 'title', 'zip_code', 'policy_code', 'addr_state', 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal', 'revol_util', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'application_type', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'tot_coll_amt', 'tot_cur_bal', 'next_pymnt_d', 'pymnt_plan', 'total_rev_hi_lim', 'sub_grade', 'emp_title'])

# %%
#missing values
missing_values = (df_1.isnull().sum()/len(df_1)*100).round(2).sort_values(ascending=False)
missing_values_df = pd.DataFrame(missing_values, columns=['Missing Values Percentage'])
missing_values_df

# %%
df_2 = df_1.drop(columns=['inq_last_12m', 'total_bal_il','verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m', 'annual_inc_joint', 'mths_since_rcnt_il', 'open_il_24m', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'dti_joint'])

# %% [markdown]
# Define Good - Bad

# %%
df_2.loan_status.value_counts(normalize=True)*100

# %%
def categorize_loan(row):
    if row['loan_status'] in ['Fully Paid', 'Current', 'In Grace Period', 'Does not meet the credit policy. Status:Fully Paid']:
        return 0  # Good loan
    else:
        return 1  # Bad loan

# Apply the function to create the loan_category feature
df_2['loan_category'] = df_2.apply(categorize_loan, axis=1)

# %%
# Count the number of loans in each category
category_counts = df_2['loan_category'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=['Good Loans', 'Bad Loans'], autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Loan Categories')
plt.show()

# %% [markdown]
# Exploratory Data Analysis

# %%
# split numerical and categorical
categorical_col = []
numerical_col = []

for i in df_2.columns:
  if new_df[i].dtype == 'object':
    categorical_col.append(i)
  else:
    numerical_col.append(i)

print('Categorical Columns')
print(categorical_col)
print(len(categorical_col))
print('Numerical Columns')
print(numerical_col)
print(len(numerical_col))

# %%
df_2[categorical_col].describe().transpose()

# %%
df_2[numerical_col].describe().transpose()

# %%
# Univariate

# %%
#delinq2yrs
plt.figure(figsize=(8, 6))
sns.countplot(data=df_2, x='delinq_2yrs', order=df['delinq_2yrs'].value_counts().index)
plt.title('Distribution of Delinquency in the Past 2 Years')
plt.xlabel('Number of 30+ Days Past-Due Incidences')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# %%
#top 3 purposes
top_purposes = df_2['purpose'].value_counts().nlargest(3).index
filtered_df_2 = df_2[df_2['purpose'].isin(top_purposes)]

# Plot distribution of top 3 loan purposes
plt.figure(figsize=(10, 6))
sns.countplot(data=filtered_df_2, x='purpose', order=top_purposes)
plt.title('Top 3 Loan Purposes')
plt.xlabel('Loan Purpose')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# %%
# loan grades
plt.figure(figsize=(8, 6))
sns.countplot(x='grade', data=df_2, palette='viridis', order=sorted(df_2['grade'].unique()))
plt.title('Distribution of Loan Grades')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()

# %%
print(categorical_col)

# %%
grade_loan_category_counts = df_2.groupby(['grade', 'loan_category']).size().unstack(fill_value=0)

# Bar chart
grade_loan_category_counts.plot(kind='bar', stacked=True)
plt.title('Distribution of Loan Category by Grade')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Loan Category', labels=['Good (0)', 'Bad (1)'])
plt.show()

# %% [markdown]
# Data Preprocessing

# %%
#fixing incorrect data type

#term
term_dtype = df_2['term'].dtype
print("Data type of 'term' column:", term_dtype)

# Convert term to integer
df_2['term'] = df_2['term'].str.extract('(\d+)').astype(int)

# Check the data type of term
print("Data type of 'term' column after conversion:", df_2['term'].dtype)


# %%
# emp_length
emp_length_dtype = df_2['emp_length'].dtype
print("Data type of 'emp_length' column:", emp_length_dtype)

# Convert emp_length to integer
df_2['emp_length'] = df_2['emp_length'].replace({'< 1 year': '0 years', '10+ years': '10 years'})
df_2['emp_length'] = df_2['emp_length'].str.extract('(\d+)').astype(float)

# Check the data type of emp_length
print("Data type of 'emp_length' column after conversion:", df_2['emp_length'].dtype)

# %%
df_2['home_ownership'] = df_2['home_ownership'].replace(['OTHER', 'ANY', 'NONE'], 'OTHER')

# %%
#missing values
missing_values_2 = (df_2.isnull().sum()/len(df_2)*100).round(2).sort_values(ascending=False)
missing_values_df_2 = pd.DataFrame(missing_values_2, columns=['Missing Values Percentage'])
missing_values_df_2

# %%
df_3 = df_2.drop(columns=['mths_since_last_record','desc'])

# %%
# split numerical and categorical
categorical_col_1 = []
numerical_col_1 = []

for i in df_3.columns:
  if new_df[i].dtype == 'object':
    categorical_col_1.append(i)
  else:
    numerical_col_1.append(i)

print('Categorical Columns')
print(categorical_col_1)
print('Numerical Columns')
print(numerical_col_1)

# %%
# fill missing values
# categorical
df_3['home_ownership'].fillna(df_3['home_ownership'].mode()[0], inplace=True)
df_3['verification_status'].fillna(df_3['verification_status'].mode()[0], inplace=True)
df_3['loan_status'].fillna(df_3['loan_status'].mode()[0], inplace=True)
df_3['purpose'].fillna(df_3['purpose'].mode()[0], inplace=True)
df_3['initial_list_status'].fillna(df_3['initial_list_status'].mode()[0], inplace=True)

# numerical
df_3['emp_length'].fillna(df_3['emp_length'].median(), inplace=True)
df_3['open_acc'].fillna(df_3['open_acc'].median(), inplace=True)
df_3['acc_now_delinq'].fillna(df_3['acc_now_delinq'].median(), inplace=True)
df_3['total_acc'].fillna(df_3['total_acc'].median(), inplace=True)
df_3['pub_rec'].fillna(df_3['pub_rec'].median(), inplace=True)
df_3['delinq_2yrs'].fillna(df_3['delinq_2yrs'].median(), inplace=True)
df_3['dti'].fillna(df_3['dti'].median(), inplace=True)
df_3['annual_inc'].fillna(df_3['annual_inc'].median(), inplace=True)


# %%
#missing values
missing_values_3 = (df_3.isnull().sum()/len(df_3)*100).round(2).sort_values(ascending=False)
missing_values_df_3 = pd.DataFrame(missing_values_3, columns=['Missing Values Percentage'])
missing_values_df_3

# %%
# feature selection

# %%
df_3[numerical_col_1].info()

# %%
df_4 = df_3.drop(columns=['loan_status'])

# %%
cat = ['grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status']

# %%
num = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'emp_length', 'annual_inc', 'dti', 'delinq_2yrs', 'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'loan_category']

# %%
# feature encoding
df_encoded = df_4.copy()
df_encoded[cat].describe().transpose()

# %%
# label encoding

# %%
from sklearn.preprocessing import LabelEncoder

# 1. Grade - Label Encoding
grade_encoder = LabelEncoder()
df_encoded['grade'] = grade_encoder.fit_transform(df_encoded['grade'])
df_encoded.head(5)


# %%
# 2. Verification Status - Label Encoding
verification_status_encoder = LabelEncoder()
df_encoded['verification_status'] = verification_status_encoder.fit_transform(df_encoded['verification_status'])
df_encoded.head(5)

# %%
# 3. Initial List Status - Label Encoding
initial_list_status_encoder = LabelEncoder()
df_encoded['initial_list_status'] = initial_list_status_encoder.fit_transform(df_encoded['initial_list_status'])
df_encoded.head(5)

# %%
# 4. home_ownership, purpose - one hot encoding
ohe_columns = ['home_ownership', 'purpose']

df_encoded_2 = pd.get_dummies(df_encoded, columns=ohe_columns, drop_first=True)

# Display the updated DataFrame
df_encoded_2.head(10)

# %%
# split features and target
X = df_encoded_2.drop(columns=['loan_category'])  # Features
y = df_encoded_2['loan_category']  # Target variable

# %%
# split train and test
from sklearn.model_selection import train_test_split

# 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# loan category for train data

a = pd.Series(y_train).value_counts().reset_index()
a.columns=['loan_category','total']
a['%'] = round(a['total']*100/sum(a['total']),3)
a

# %%
# loan category for train data
value_counts = pd.Series(y_train).value_counts()

# pie chart
plt.figure(figsize=(8, 8))
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Loan Categories for Training Data')
plt.show()

# %%
# SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# %%
plt.figure(figsize=(6, 6))
plt.pie(y_resampled.value_counts(), labels=y_resampled.value_counts().index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Loan Categories for Training Data after SMOTE')
plt.show()

# %% [markdown]
# Data Modelling

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(X_resampled, y_resampled)

# Predict on the training data
y_train_pred = logreg.predict(X_resampled)

# Evaluate performance on the training data
train_accuracy = accuracy_score(y_resampled, y_train_pred)
print("Train Accuracy:", train_accuracy)

# Predict on the test data
y_test_pred = logreg.predict(X_test)

# Evaluate performance on the test data
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# %%
# decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the model on the resampled training data
dt_classifier.fit(X_resampled, y_resampled)

# Predict on the train data
y_train_pred_dt = dt_classifier.predict(X_resampled)

# Evaluate performance on the train data
train_accuracy_dt = accuracy_score(y_resampled, y_train_pred_dt)
print("Decision Tree Train Accuracy:", train_accuracy_dt)

# Predict on the test data
y_test_pred_dt = dt_classifier.predict(X_test)

# Evaluate performance on the test data
test_accuracy_dt = accuracy_score(y_test, y_test_pred_dt)
print("Decision Tree Test Accuracy:", test_accuracy_dt)


# %%
# feature importances

feature_importances_dt = dt_classifier.feature_importances_
feature_names = X_train.columns


feature_importance_dict = dict(zip(feature_names, feature_importances_dt))

sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("Feature Importances:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")

# %%
rev_sorted_feature_importance = sorted_feature_importance[::-1]
features = [feature[0] for feature in rev_sorted_feature_importance]
importances = [feature[1] for feature in rev_sorted_feature_importance]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


