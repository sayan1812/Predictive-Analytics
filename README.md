# Predictive-Analytics
new repo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the data
df = pd.read_csv('data.csv')

# Preprocess the data
X = df.drop(['target'], axis=1)
y = df['target']

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Encode categorical variables
X = pd.get_dummies(X, columns=['sex', 'diet', 'exercise', 'smoking'])

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature engineering
rfe = RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=10)
rfe.fit(X_scaled, y)
X_selected = rfe.transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Develop the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Deploy the model
def predict_disease_onset(patient_data):
    patient_data = pd.DataFrame([patient_data])  # Convert to DataFrame with one row
    patient_data.fillna(patient_data.mean(), inplace=True)
    patient_data = pd.get_dummies(patient_data, columns=['sex', 'diet', 'exercise', 'smoking'])
    patient_data = scaler.transform(patient_data)
    patient_data = rfe.transform(patient_data)
    prediction = model.predict(patient_data)
    return prediction

# Example usage
patient_data = {'age': 35, 'sex': 'male', 'diet': 'healthy', 'exercise': 'regular', 'smoking': 'non-smoker', 'medical_history': ['diabetes'], 'lab_results': [120, 80], 'medication_list': ['metformin']}
patient_data = pd.DataFrame([patient_data])  # Convert to DataFrame with one row
patient_data['medical_history'] = patient_data['medical_history'].apply(lambda x: ','.join(x))  # Convert list to string
patient_data['lab_results'] = patient_data['lab_results'].apply(lambda x: ','.join(map(str, x)))  # Convert list to string
patient_data['medication_list'] = patient_data['medication_list'].apply(lambda x: ','.join(x))  # Convert list to string
prediction = predict_disease_onset(patient_data)
print('Likelihood of disease onset:', prediction)
