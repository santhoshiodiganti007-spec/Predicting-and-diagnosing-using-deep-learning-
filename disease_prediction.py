# ==========================================
# Predicting and Diagnosing Diseases Using Deep Learning
# ==========================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Step 2: Load Dataset
data = pd.read_csv("Training.csv")
# Remove unwanted column if exists
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Step 3: Separate Features and Target
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Step 4: Encode Target Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Step 6: Build Deep Learning Model
model = Sequential()

model.add(Dense(256, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))

model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Step 7: Compile Model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Step 8: Train Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Step 9: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Step 10: Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Step 11: Predict New Patient Symptoms
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)
predicted_disease = le.inverse_transform([np.argmax(prediction)])

print("Predicted Disease:", predicted_disease[0])