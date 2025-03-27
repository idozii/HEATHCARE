import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

healthcare_filepath = "data/healthcare_dataset.csv"
data = pd.read_csv(healthcare_filepath)

#Int, Float: Age, Billing Amount, Room Number
#No null
#Handle: Name, Doctor, Date of Admission, Discharge Date

data['Name'] = data['Name'].str.strip() 
data['Name'] = data['Name'].str.title()

data['Doctor'] = data['Doctor'].str.strip()
data['Doctor'] = data['Doctor'].str.title()

data['Date of Admission'] = pd.to_datetime(data['Date of Admission'])

data['Discharge Date'] = pd.to_datetime(data['Discharge Date'])

data['Length of Stay'] = (data['Discharge Date'] - data['Date of Admission']).dt.days

plt.figure(figsize=(10,6))
plt.title("Length of Stay Distribution")
plt.hist(data['Length of Stay'], bins=20, edgecolor='black')
plt.xlabel('Days')
plt.savefig('length_of_stay.png')

plt.figure(figsize=(12,6))
plt.title("Average Billing Amount by Age Group")
age_groups = pd.cut(data['Age'], bins=[0, 18, 35, 50, 65, 100])
avg_billing = data.groupby(age_groups)['Billing Amount'].mean()
avg_billing.plot(kind='bar')
plt.ylabel('Average Billing Amount ($)')
plt.savefig('billing_by_age.png')

# Target: Test Results

X = data.drop(['Test Results'], axis=1)
y = data['Test Results']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))
print('Accuracy:', accuracy_score(y_valid, preds))

