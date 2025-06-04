
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
#from tensorflow.keras import layers, Model

def unet_model_with_pooling(output_channels: int):
    """
    U-Net model with pooling layers using Functional API with Depthwise and Pointwise convolutions.

    Args:
        output_channels: Number of output channels (e.g., 3 for RGB images).

    Returns:
        A tf.keras.Model representing the U-Net with pooling.
    """
    inputs = layers.Input(shape=[512, 512, 3])
    initializer = tf.random_normal_initializer(0., 0.02)
    skips = []

    x = inputs

    # Downsampling path parameters (filters, kernel_size, apply_batchnorm)
    down_params = [
        (64, 3, False),  # First two blocks without BatchNorm
        (64, 3, False),
        (128, 3, True),
        (256, 3, True),
        (512, 3, True),
        (512, 3, True),
        (512, 3, True),
    ]

    # Build the downsampling path
    for filters, size, apply_bn in down_params:
        # Depthwise Convolution
        x = layers.DepthwiseConv2D(
            size, strides=1, padding='same',
            depthwise_initializer=initializer, use_bias=False
        )(x)
        # Pointwise Convolution
        x = layers.Conv2D(
            filters, 1, strides=1, padding='same',
            kernel_initializer=initializer, use_bias=False
        )(x)
        if apply_bn:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        # MaxPooling reduces spatial dimensions
        x = layers.MaxPooling2D(2, strides=2, padding='same')(x)
        skips.append(x)  # Save the output for skip connections

    # Upsampling path parameters (filters, kernel_size)
    up_params = [
        (512, 3), (512, 3), (512, 3),
        (256, 3), (128, 3), (64, 3), (64, 3)
    ]

    cnt = len(skips) - 2  # Skip the bottleneck layer

    # Build the upsampling path
    for filters, size in up_params:
        # Transposed Convolution for upsampling
        x = layers.Conv2DTranspose(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        # Add skip connection if available
        if cnt >= 0:
            x = layers.Concatenate()([x, skips[cnt]])
            cnt -= 1

    # Final output layer
    x = layers.Conv2D(
        output_channels, 1, padding='same', activation='sigmoid'
    )(x)

    return Model(inputs=inputs, outputs=x)
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
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))

    # Define parameters
    batch_size = 4
    epochs =100
    output_channels = 3  # RGB images
    output_dir = './output'

    # Prepare datasets
    train_dataset = prepare_dataset(input_dir, gt_dir,batch_size)
    #val_dataset = prepare_dataset_from_directory(val_dir, batch_size, shuffle=False)
    
    # Initialize the U-Net model
    model = unet_model_with_pooling(output_channels)

    # Train the model
    history = train_model(model, train_dataset, epochs, batch_size, output_dir)
    model.save("AdvancedSSIM.h5")
    print("Final model saved.")
