import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Importing data
data = pd.read_csv(r'C:\Users\danis\Downloads\archive-Telco-customer-churn-F\Assets\WA_Fn-UseC_-Telco-Customer-Churn.csv', index_col=False)
eda = pd.read_csv(r'C:\Users\danis\Downloads\archive-Telco-customer-churn-F\clean_test3.csv')
categorical_features = []
numerical_features = []
for col in eda.columns:
    if len(eda[col].unique()) <= 10:
        categorical_features.append(col)
    else:
        numerical_features.append(col)
	
st.sidebar.header('Telco Customer Churn Analysis')
menu = st.sidebar.radio(
    "Menu:",
    ("Intro", "Data", "Analysis", "Models"),
)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Project Submitted By: Danish Rasheed (VR497604)')
st.sidebar.write('Github Repositories:')
st.sidebar.write('https://github.com/D-Rasheed/programming_project_UNIVR/tree/main')
if menu == 'Intro':
   st.write('The project discuss a comprehensive concept about churn analysis: Churn analysis in data analytics examines customer attrition by analyzing patterns and factors that contribute to customer churn, helping businesses retain valuable customers and improve their strategies.')
   st.write('Content Each row represents a client, and each column has the properties mentioned in the column Metadata.')
   st.write('The data collection contains information about:')
   st.write('Customers that departed during the previous month the column is called Churn Services that each customer has signed up for phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.')
   st.write('Customer account details how long they have been a customer, contract, payment method, paperless billing, monthly costs, and total charges.')
   st.write('Customers demographic information, such as gender, age range, and whether or not they have partners or dependents.')
	
elif menu == 'Data':

   st.title("DataFrame:")
   st.write(">***7043 entries | 21 columns***")
   st.dataframe(data)
elif menu == 'Analysis':
   st.subheader("Distribution of Numerical Features")
   fig, axs = plt.subplots(1, 3, figsize=(15, 4))
   axs[0].hist(eda['tenure'], bins=30)
   axs[0].set_title('Tenure')
   axs[1].hist(eda['MonthlyCharges'], bins=30)
   axs[1].set_title('Monthly Charges')
   axs[2].hist(eda['TotalCharges'], bins=30)
   axs[2].set_title('Total Charges')
   st.pyplot(fig)
   # Countplot of categorical features
   st.subheader("Countplot of Categorical Features")
   fig, axs = plt.subplots(1, 5, figsize=(20, 5))
   sns.countplot(data=eda, x='Churn', hue='gender', ax=axs[0])
   sns.countplot(data=eda, x='Churn', hue='SeniorCitizen', ax=axs[1])
   sns.countplot(data=eda, x='Churn', hue='Partner', ax=axs[2])
   sns.countplot(data=eda, x='Churn', hue='Dependents', ax=axs[3])
   sns.countplot(data=eda, x='Churn', hue='PhoneService', ax=axs[4])
   st.pyplot(fig)
   # Boxplot of numerical features against the target column
   st.subheader("Boxplot of Numerical Features against Churn")
   fig, axs = plt.subplots(1, 3, figsize=(15, 4))
   sns.boxplot(data=eda, x='Churn', y='tenure', ax=axs[0])
   sns.boxplot(data=eda, x='Churn', y='MonthlyCharges', ax=axs[1])
   sns.boxplot(data=eda, x='Churn', y='TotalCharges', ax=axs[2])
   st.pyplot(fig)
   # Scatterplots of numerical features with hue='Churn'
   st.subheader("Scatterplots of Numerical Features with Churn")
   fig, axs = plt.subplots(1, 3, figsize=(15, 4))
   sns.scatterplot(data=eda, x='MonthlyCharges', y='TotalCharges', hue='Churn', ax=axs[0])
   sns.scatterplot(data=eda, x='tenure', y='MonthlyCharges', hue='Churn', ax=axs[1])
   sns.scatterplot(data=eda, x='tenure', y='TotalCharges', hue='Churn', ax=axs[2])
   st.pyplot(fig)

elif menu == 'Models':
   # Data preprocessing
    df_final = eda.copy()
    le = LabelEncoder()

    for f in categorical_features:
        df_final[f] = le.fit_transform(df_final[f])

    df_final['Churn'] = le.fit_transform(df_final['Churn'])
    

    x = df_final.drop('Churn', axis=1)
    y = df_final['Churn']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

    # Model training and evaluation

    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    pred_rf = rf.predict(x_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)
    st.write("Random Forest Mean Absolute Error:", mae_rf)

