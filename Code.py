import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.optimizers import Adam
import seaborn as sns

# File path
file_path = "C:/Users/shalu/Downloads/Vehicle tyre.csv"

# Load the data
df = pd.read_csv(file_path)
df = df.fillna(0)

# Convert price columns to float
df['Selling Price'] = df['Selling Price'].str.replace(',', '').astype(float)
df['Original Price'] = df['Original Price'].str.replace(',', '').astype(float)

# Extract Size column
df[['Width', 'Aspect_Ratio', 'Diameter']] = df['Size'].str.extract(r'(\d+)/(\d+) R (\d+)').astype(float)
df = df.drop(columns=['Size'])

# Convert categorical columns to strings
categorical_columns = ['Brand', 'Model', 'Submodel', 'Tyre Brand', 'Serial No.', 'Type']
for column in categorical_columns:
    df[column] = df[column].astype(str)

# Encode categorical columns
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Encode the target variable
y = df['Rating']
label_encoders['Rating'] = LabelEncoder()
y = label_encoders['Rating'].fit_transform(y)
y = to_categorical(y)

# Prepare features
X = df.drop(columns=['Rating'])
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model with 15 epochs
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nModel Evaluation:\nLoss: {loss:.4f}\nAccuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('car_tyre_classifier_model.keras')
print("\nModel saved to 'car_tyre_classifier_model.keras'")

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate precision
precision = precision_score(y_test_classes, y_pred_classes, average='weighted', zero_division=1)
print(f"\nPrecision: {precision * 100:.2f}%")

# Select 10 vehicles from the dataset
selected_vehicles = df.sample(n=10, random_state=42)
X_selected = scaler.transform(selected_vehicles.drop(columns=['Rating']))

# Predict ratings for the selected vehicles
predicted_ratings = model.predict(X_selected)
predicted_classes = np.argmax(predicted_ratings, axis=1)

# Add predictions to the selected vehicles
selected_vehicles['Predicted_Rating'] = predicted_classes

# Find the best vehicle index
best_vehicle_index = selected_vehicles['Predicted_Rating'].idxmax()

# Decode and print the best model name and tyre brand
best_model_name = label_encoders['Model'].inverse_transform([selected_vehicles.loc[best_vehicle_index, 'Model']])[0]
best_tyre_brand_name = label_encoders['Tyre Brand'].inverse_transform([selected_vehicles.loc[best_vehicle_index, 'Tyre Brand']])[0]
print(f"\nBest model among the selected 10: {best_model_name}")
print(f"Best tyre brand among the selected 10: {best_tyre_brand_name}")

# Plot the model loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

