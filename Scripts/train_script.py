import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Enable mixed precision training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    gpu_device = physical_devices[0]
    try:
        tf.config.set_logical_device_configuration(
            gpu_device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]  # Set memory limit in MB if needed
        )
        print("GPU is available and configured with memory limits.")
    except RuntimeError as e:
        print("Error configuring GPU memory:", e)
else:
    print("No GPU detected. Falling back to CPU.")

# Load preprocessed data
print("Loading preprocessed dataset...")
data = pd.read_csv('Processed_dataset.csv')  # Ensure preprocessed dataset is saved as 'preprocessed_data.csv'

# Verify dataset
if data.isnull().sum().any():
    print("Dataset contains missing values. Applying dynamic imputation.")
    data.fillna(data.mean(), inplace=True)  # Impute missing values with the mean of each column

# Ensure the dataset only contains numeric columns
print("Filtering numeric columns in the dataset...")
data = data.select_dtypes(include=[float, int])

# Split data into training and validation sets
print("Splitting dataset into training and validation sets...")
X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)

# Input dimensions
input_dim = X_train.shape[1]
print(f"Input dimension for the model: {input_dim}")

# Define the autoencoder model
def create_autoencoder(encoding_dim=16, hidden_dims=[128, 64], dropout_rate=0.2):
    """
    Create an autoencoder with parameterized layer dimensions and dropout rate.

    Parameters:
    - encoding_dim: int, dimension of the bottleneck layer.
    - hidden_dims: list of int, dimensions of hidden layers.
    - dropout_rate: float, dropout rate for regularization.

    Returns:
    - autoencoder: keras Model, the compiled autoencoder model.
    """
    print("Building autoencoder model...")
    input_layer = Input(shape=(input_dim,))
    encoded = input_layer
    for dim in hidden_dims:
        print(f"Adding encoding layer with {dim} units.")
        encoded = Dense(dim, activation='relu')(encoded)
        encoded = Dropout(dropout_rate)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    print("Adding bottleneck layer.")

    decoded = encoded
    for dim in reversed(hidden_dims):
        print(f"Adding decoding layer with {dim} units.")
        decoded = Dense(dim, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    print("Adding output layer.")

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder

# Initialize autoencoder
autoencoder = create_autoencoder(encoding_dim=16, hidden_dims=[128, 64], dropout_rate=0.2)

# Compile the model
print("Compiling the autoencoder model...")
autoencoder.compile(optimizer='adam', loss='mse')

# Define callbacks
class LoggingLearningRateCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        print(f"Epoch {epoch + 1}: Learning rate is {lr}")

logging_lr_callback = LoggingLearningRateCallback()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='best_autoencoder.keras', monitor='val_loss', save_best_only=True, verbose=1)

# Create TensorFlow datasets for optimized performance
print("Creating TensorFlow datasets for training and validation...")
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
train_dataset = train_dataset.shuffle(2048).batch(256).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, X_val)).batch(256)

# Train the model
print("Starting training...")
history = autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, reduce_lr, model_checkpoint, logging_lr_callback]
)

# Plot the training and validation loss
print("Plotting training and validation loss...")
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Loss)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Perform predictions
print("Performing predictions on the dataset...")
predictions = autoencoder.predict(data)
print("Predictions completed.")

# Calculate reconstruction error
print("Calculating reconstruction error...")
data['Reconstruction_Error'] = ((data - predictions) ** 2).mean(axis=1)
print("Reconstruction error calculation completed.")

# Define anomaly threshold
threshold = data['Reconstruction_Error'].quantile(0.95)  # Top 5% of reconstruction errors
print(f"Anomaly threshold set at: {threshold}")

# Flag anomalies based on threshold
print("Flagging anomalies...")
data['Anomaly_Flag'] = (data['Reconstruction_Error'] > threshold).astype(int)
print("Anomaly flagging completed.")

# Save the updated dataset
output_file = 'data_with_anomalies.csv'
print(f"Saving updated dataset with anomalies to {output_file}...")
data.to_csv(output_file, index=False)
print(f"Updated dataset saved successfully to {output_file}.")
