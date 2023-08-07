import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from skimage.color import rgb2gray

# Define the generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=100))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(np.prod(image_size), activation='tanh'))
    model.add(layers.Reshape(image_size))
    return model

# Define the discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=image_size))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


if  __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Define the directory containing your images
    data_dir = 'data'

    # Define the size of your input images
    image_size = (64, 64)

    # Prepare the dataset
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
    num_images = len(image_files)

    # Load and preprocess the images
    images = []
    for image_file in image_files:
        image_path = os.path.join(data_dir, image_file)
        image = rgb2gray(Image.open(image_path).resize(image_size))
        #image = np.array(image) / 255.0
        images.append(image)
    images = np.array(images)

    # Define the generator and discriminator
    generator = build_generator()
    discriminator = build_discriminator()

    # Define the GAN model
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(100,))
    gan_output = discriminator(generator(gan_input))
    gan = tf.keras.Model(gan_input, gan_output)

    # Compile the models
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))
    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    # Training loop
    epochs = 100
    batch_size = 10
    steps_per_epoch = num_images // batch_size

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # Generate random noise as input to the generator
            noise = np.random.normal(0, 1, size=(batch_size, 100))

            # Generate fake images
            generated_images = generator.predict(noise)

            # Select a random batch of real images
            real_images = images[np.random.randint(0, num_images, size=batch_size)]

            # Combine real and fake images into a single batch
            batch_images = np.concatenate([real_images, generated_images])

            # Create labels for the discriminator
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

            # Train the discriminator
            discriminator_loss = discriminator.train_on_batch(batch_images, labels)

            # Train the generator
            noise = np.random.normal(0, 1, size=(batch_size, 100))
            generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print the progress every epoch
        print(f"Epoch {epoch+1}/{epochs} - Discriminator Loss: {discriminator_loss} - Generator Loss: {generator_loss}")

        # Generate and save a sample of generated images
        if (epoch+1) % 10 == 0:
            noise = np.random.normal(0, 1, size=(5, 100))
            generated_images = generator.predict(noise) * 0.5 + 0.5
            for i in range(5):
                plt.imshow(generated_images[i])
                plt.axis('off')
                plt.savefig(f"data/output/generated_image_{epoch+1}_{i+1}.png")
                plt.close()