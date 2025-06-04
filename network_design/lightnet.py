
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Conv2D, Activation
from tensorflow.keras.models import Model

@tf.keras.utils.register_keras_serializable()
def ssim_loss(y_true, y_pred):
    """
    SSIM Loss for image quality enhancement.
    Args:
        y_true: Ground truth images.
        y_pred: Predicted images.
    Returns:
        A scalar loss value based on SSIM.
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1.0 - tf.reduce_mean(ssim)

def build_1x1_cnn_functional(
    input_shape=(512, 512, 3),
    output_channels=3,
    hidden_channels=64,
    num_hidden_layers=3
):
    # Input tensor
    input_layer = Input(shape=input_shape, name='input_image')
    
    # Initial feature expansion
    x = Conv2D(hidden_channels, (1, 1), name='expand_channels')(input_layer)
    x = Activation('relu', name='initial_relu')(x)
    
    # Hidden layers with channel mixing
    for i in range(num_hidden_layers):
        x = Conv2D(hidden_channels, (1, 1), name=f'hidden_{i}')(x)
        x = Activation('relu', name=f'relu_{i}')(x)
        # Optional: Add BatchNorm after convolution
        # x = BatchNormalization()(x)
    
    # Final output projection
    output_layer = Conv2D(output_channels, (1, 1), name='output_projection')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer, name='1x1_cnn')
    
    return model
def load_and_preprocess_image(file_path):
    # Ensure the file path is correctly handled as a string
    image = tf.io.read_file(tf.strings.join([file_path]))  # Concatenate to ensure string type
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0
    return image
def prepare_dataset(input_dir, gt_dir, batch_size=32):
    # Create file paths and ensure they are handled as strings
    input_images = [os.path.join(input_dir, fname) for fname in sorted(os.listdir(input_dir))]
    gt_images = [os.path.join(gt_dir, fname) for fname in sorted(os.listdir(gt_dir))]

    # Convert lists to TensorFlow constants to enforce the dtype as string
    input_images = tf.constant(input_images, dtype=tf.string)
    gt_images = tf.constant(gt_images, dtype=tf.string)

    # Create Dataset objects for input and ground truth
    input_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(load_and_preprocess_image)
    gt_dataset = tf.data.Dataset.from_tensor_slices(gt_images).map(load_and_preprocess_image)

    # Pair and batch the datasets
    dataset = tf.data.Dataset.zip((input_dataset, gt_dataset))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# 6. Training the model
def train_model(model, train_dataset, epochs):
    """
    Trains the model with a given dataset and saves the results.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Compile the model with the hybrid loss function
    model.compile(optimizer=optimizer, loss=ssim_loss, metrics=['mae'])

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs
    )

    return history

    # Define parameters
batch_size = 8
epochs = 100
output_channels = 3  # RGB images
output_dir = './output'
input_dir = 'MasterNetverk/LSUI/input'  # Path to training images
gt_dir = 'MasterNetverk/LSUI/GT' 
# Prepare datasets
train_dataset = prepare_dataset(input_dir, gt_dir,batch_size)


# Create and verify the model
model = build_1x1_cnn_functional(
    input_shape=(512, 512, 3),
    output_channels=3,
    hidden_channels=64,
    num_hidden_layers=3
)

history= train_model(model, train_dataset, epochs)

model.save("1x1CNNSSIM.h5")

