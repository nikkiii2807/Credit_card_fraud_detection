# app/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_and_save_model():
    # Load the dataset
    data = pd.read_csv('../datasets/new_dataset.csv')  # Adjust the path as needed

    # Select relevant features
    X = data[['Time', 'Amount']]
    y = data['Class']  # Assuming 'Class' is the target variable

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model as a pickle file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Run this function to train and save the model
if __name__ == "__main__":
    train_and_save_model()