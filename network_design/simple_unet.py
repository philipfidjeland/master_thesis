import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model

def downsample(filters, size, apply_activation=True):
    """Creates a downsampling block with Conv2D and LeakyReLU."""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = [
        layers.Conv2D(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    ]
    if apply_activation:
        result.append(layers.ReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    """Creates an upsampling block with Conv2DTranspose and LeakyReLU."""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = [
        layers.Conv2DTranspose(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    ]
    if apply_dropout:
        result.append(layers.Dropout(0.5))
    result.append(layers.ReLU())
    return result

def build_unet(output_channels):
    """Builds the U-Net model using the Functional API without Sequential layers."""
    # Input layer
    inputs = layers.Input(shape=[512, 512, 3])  # Adjust input shape as needed

    # Downsampling path
    # Block 1
    x = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(inputs)
    x = layers.ReLU()(x)
    skip1 = x  # Save skip connection

    # Block 2
    x = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    x = layers.ReLU()(x)
    skip2 = x  # Save skip connection

    # Block 3
    x = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    x = layers.ReLU()(x)
    skip3 = x  # Save skip connection

    # Block 4
    x = layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    x = layers.ReLU()(x)

    # Upsampling path with skip connections
    # Block 5
    x = layers.Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Concatenate()([x, skip3])  # Skip connection from Block 3

    # Block 6
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Concatenate()([x, skip2])  # Skip connection from Block 2

    # Block 7
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip1])  # Skip connection from Block 1

    # Block 8
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(x)
    x = layers.ReLU()(x)

    # Final output layer
    outputs = layers.Conv2D(
        output_channels, (1, 1), padding='same', activation='sigmoid'
    )(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 2. Define SSIM loss function
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

@tf.keras.utils.register_keras_serializable()
def hybrid_loss(y_true, y_pred):
    """
    Hybrid loss combining SSIM and MAE.
    Args:
        y_true: Ground truth images.
        y_pred: Predicted images.
    Returns:
        A scalar loss value combining SSIM and MAE.
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = 1.0 - tf.reduce_mean(ssim)
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return ssim_loss + mae_loss



# 5. Dataset preparation (for training)
def load_image(image_file):
    """
    Loads and preprocesses an input image.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image to [0, 1]
    return image
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
def train_model(model, train_dataset, epochs, batch_size, output_dir):
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



if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    # Define paths to your training and validation directories
    input_dir = 'MasterNetverk/LSUI/input'  # Path to training images
    gt_dir = 'MasterNetverk/LSUI/GT'      # Path to validation images


    # Define parameters
    batch_size = 4
    epochs = 100
    output_channels = 3  # RGB images
    output_dir = './output'

    # Prepare datasets
    train_dataset = prepare_dataset(input_dir, gt_dir,batch_size)
    #val_dataset = prepare_dataset_from_directory(val_dir, batch_size, shuffle=False)
    

    model = build_unet(output_channels)

    # Train the model
    history = train_model(model, train_dataset, epochs, batch_size, output_dir)
    model.save("SimpleSSIM.h5")

    print("Final model saved.")
