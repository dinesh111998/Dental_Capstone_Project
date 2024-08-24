from data_preprocessing import load_and_preprocess_data
from eda import perform_eda
from model_building import build_and_evaluate_models
from sklearn.model_selection import train_test_split


X, y = load_and_preprocess_data('../data/Dentistry Dataset.csv')


perform_eda(pd.DataFrame(X))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


build_and_evaluate_models(X_train, X_test, y_train, y_test)