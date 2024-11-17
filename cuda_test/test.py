import numpy as np
import pandas as pd
from tensorflow.keras.layers import ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# Check GPU availability
if not tf.config.list_physical_devices('GPU'):
    print("No GPU detected. The model will run on the CPU.")
else:
    print("GPU detected. The model will run on the GPU.")

# Load the data
data = pd.read_csv(r'C:\Users\Malub.000\.spyder-py3\AI_project_alpha\RNN_coursework\data_set\dataset2.csv')  # Update with your CSV file path
username = 'zg819-2'  # Replace with actual username

# Split features (X) and target (y)
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Only the last column (Target hit)

# Normalize the feature data for better performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaling parameters
offset = scaler.mean_  # Mean of each feature
scale = scaler.scale_  # Standard deviation of each feature
scaling_params = np.vstack([offset, scale])  # Create 2 x 6 array

# Save scaling parameters to a file
np.savetxt(rf'C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\cuda_test\{username}.txt', scaling_params, delimiter=' ') 

# Convert y to categorical (two classes: 0 and 1)
y_binary = to_categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))

model.add(Dense(32))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))

model.add(Dense(16))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.1))

model.add(Dense(8))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(2, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)

# Train the model inside a GPU scope
with tf.device('/GPU:0'):  # Explicitly run on GPU if available
    history = model.fit(X_train, y_train, epochs=1250, batch_size=50, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])

# Save the model
model.save(rf'C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\cuda_test\{username}.keras')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
