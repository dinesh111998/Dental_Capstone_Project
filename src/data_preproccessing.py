import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data("C:\Users\manis\Downloads\project data and file\Dentistry Dataset.csv"):
    data = pd.read_csv("C:\Users\manis\Downloads\project data and file\Dentistry Dataset.csv")
    
    
    data.fillna(data.mean(), inplace=True)
    
    
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    
    
    X = data.drop(['Gender'], axis=1)
    y = data['Gender']
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y