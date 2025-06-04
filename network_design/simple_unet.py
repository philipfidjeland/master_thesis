def build_unet(output_channels):
    """Builds the 6-layer U-Net model using Functional API without dropout."""
    # Input layer
    inputs = layers.Input(shape=[512, 512, 3])

    # Downsampling path (6 layers)
    # Block 1: 512x512 -> 256x256
    x = layers.Conv2D(64, 4, strides=2, padding='same', 
                      kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                      use_bias=False)(inputs)
    x = layers.ReLU()(x)
    skip1 = x

    # Block 2: 256x256 -> 128x128
    x = layers.Conv2D(128, 4, strides=2, padding='same', 
                      kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                      use_bias=False)(x)
    x = layers.ReLU()(x)
    skip2 = x

    # Block 3: 128x128 -> 64x64
    x = layers.Conv2D(256, 4, strides=2, padding='same', 
                      kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                      use_bias=False)(x)
    x = layers.ReLU()(x)
    skip3 = x

    # Block 4: 64x64 -> 32x32
    x = layers.Conv2D(512, 4, strides=2, padding='same', 
                      kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                      use_bias=False)(x)
    x = layers.ReLU()(x)
    skip4 = x

    # Block 5: 32x32 -> 16x16
    x = layers.Conv2D(512, 4, strides=2, padding='same', 
                      kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                      use_bias=False)(x)
    x = layers.ReLU()(x)
    skip5 = x

    # Block 6 (bottleneck): 16x16 -> 8x8
    x = layers.Conv2D(512, 4, strides=2, padding='same', 
                      kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                      use_bias=False)(x)
    x = layers.ReLU()(x)

    # Upsampling path (6 layers with skip connections)
    # Block 1: 8x8 -> 16x16 (with skip5)
    x = layers.Conv2DTranspose(512, 4, strides=2, padding='same', 
                               kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                               use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip5])  # Skip connection without dropout

    # Block 2: 16x16 -> 32x32 (with skip4)
    x = layers.Conv2DTranspose(512, 4, strides=2, padding='same', 
                               kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                               use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip4])  # Skip connection without dropout

    # Block 3: 32x32 -> 64x64 (with skip3)
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same', 
                               kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                               use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip3])  # Skip connection without dropout

    # Block 4: 64x64 -> 128x128 (with skip2)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', 
                               kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                               use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip2])

    # Block 5: 128x128 -> 256x256 (with skip1)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', 
                               kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                               use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip1])

    # Block 6: 256x256 -> 512x512 (final upsampling)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', 
                               kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                               use_bias=False)(x)
    x = layers.ReLU()(x)

    # Final output layer
    outputs = layers.Conv2D(
        output_channels, (1, 1), padding='same', activation='sigmoid'
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
