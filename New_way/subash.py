import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Step 1: Load and preprocess the data
data = pd.read_csv('/home/indian/Pictures/MiniProject-main/New_way/breast_cancer_data.csv')  # Replace with the path to your dataset file

# Separate features and labels
X = data.iloc[:, 1:].values  # Features (excluding the first column)
y = data.iloc[:, 0].values  # Labels (assuming the first column is the target variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Build the deep learning model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Step 3: Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Step 4: Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
