import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

filePath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(filePath)


pd.set_option('display.max_rows',20)

#data['TotalCharges'].replace(' ', np.nan, inplace=True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
data['Churn'] = data['Churn'].map({'Yes': True, 'No': False})



print(data.dtypes)


data.boxplot(column='MonthlyCharges')
plt.show()


lower_limit = data['MonthlyCharges'].quantile(0.05)
upper_limit = data['MonthlyCharges'].quantile(0.95)
dt = data[(lower_limit > data['MonthlyCharges']) & (data['MonthlyCharges'] < upper_limit)]


print(dt)


#data = pd.get_dummies(data, columns=['MonthlyCharges'])
scaler = StandardScaler()
dt[['MonthlyCharges']] = scaler.fit_transform(dt[['MonthlyCharges']]) 
data[['tenure']] = scaler.fit_transform(data[['tenure']])
data[['TotalCharges']] = scaler.fit_transform(data[['TotalCharges']])
data[['SeniorCitizen']] = scaler.fit_transform(data[['SeniorCitizen']])

numerical_df = data.select_dtypes(include=['number'])

correlation_matrix = numerical_df.corr()

print(correlation_matrix)
 
X = dt.drop('MonthlyCharges', axis=1)
y = dt['MonthlyCharges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(10,6))
plt.scatter(data.index, data['MonthlyCharges'])
plt.title('Monthly Charges Distribution')
plt.xlabel('Index')
plt.ylabel('Monthly Charges')
plt.show()

"""
plt.figure(figsize=(10, 6))
data['Churn'].value_counts().plot(kind='bar', color='salmon')
plt.title('Churn Count')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()
"""

X = data[['tenure', 'MonthlyCharges', 'TotalCharges','SeniorCitizen']]
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Eğitim Özellikleri:")
print(X_train)
print("\nTest Özellikleri:")
print(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


#model = LogisticRegression(random_state=42)
#model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel Coefficients:")
print(model.coef_)

print("\nModel Intercept:")
print(model.intercept_)

print("XGBoost")

