# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:24:21 2024

Generative Adversarial Network (GAN) utilizando TensorFlow y Keras, para 
generar nuevas imágenes de forma autónoma.

@author: Luis A. García
"""
# Importando librerías necesarías
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization,LeakyReLU, Reshape, Conv2DTranspose, Dropout, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import matplotlib.pyplot as plt
#%% Verificando disponibilidad de GPU (opcional)
#tf.config.list_physical_devices('GPU')
#tf.test.is_gpu_available()
#import os
#os.chdir('D:\\6. NEXER\\master\\Contenido\\6. Deep Learning\\tareas')
#%% Construcción del modelo generador
"""
El generador intentará crear imágenes que parezcan reales y engañen al discriminador
"""
def make_generator_model():
    # creando capa de entrada que espera vectores aleatorio de ruido de 100 dimensiones
    noise = Input(shape=(100,))
    # creando capa densa que toma la entrada de ruido y la transforma en un vector de 7*7*256 valores, sin usar bias
    x = Dense(7*7*256, use_bias=False)(noise)
    # normalizando la salida 
    x = BatchNormalization()(x)
    # aplicando función de activación LeakyReLU, que permite que los valores negativos pequeños pasen a través de la red
    x = LeakyReLU()(x)
    # Reorganizando el vector de salida en una forma de tensor de 3D (7x7x256).
    x = Reshape((7, 7, 256))(x)
    # Aplica una capa de convolución transpuesta (deconvolución) que expande la resolución 
    # espacial de la imagen
    x = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    # Normalizando la salida y aplicando activación LeakyReLU
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # aplicando otra capa de convolución transpuesta
    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    # Normalizando plicando activación LeakyReLU
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # aplicando otra capa de convolución transpuesta
    x = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
    model = Model(inputs=noise, outputs=x)
    return model
#%% Construyendo modelo del discriminador
"""
 El discriminador se utiliza para clasificar las imágenes como reales
 (provenientes del conjunto de datos de entrenamiento) o falsas (generadas por el generador)
"""
def make_discriminator_model():
    # Creando una capa de entrada que espera imágenes de tamaño 28x28 con 1 canal (en escala de grises).
    image = Input(shape=(28, 28, 1))
    # Aplicando capa convolución 2D con 64 filtros, un tamaño de kernel de 5x5, 
    # pasos de 2 en cada dirección y relleno ('padding') igual a 'same' para mantener el tamaño espacial
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(image)
    # Aplicando función de activación LeakyReLU
    x = LeakyReLU()(x)
    # aplicando regularización Dropout que apaga el 30% de las neuronas en cada iteración
    x = Dropout(0.3)(x)
    # aplicando otra capa de convolución con 128 filtros
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    # regularización dropout que apaga el 30% de las neuronas en cada iteración
    x = Dropout(0.3)(x)
    # aplanando tensor resultante
    x = Flatten()(x)
    # aplicando capa densa
    x = Dense(1)(x)
    # Creando el modelo de Keras especificando que las entradas serán las imágenes 
    # y las salidas serán los puntajes de clasificación (real o falsa)
    model = Model(inputs=image, outputs=x)
    return model
#%% Creando función para generar y guardar imágenes.
def generate_and_save_images(model, epoch, test_input):
    # generando imagenes a partir de la entrada de prueba (test_input). 
    # El parámetro training=False indica que el modelo está en modo de inferencia, no de entrenamiento.
    predictions = model(test_input, training=False)
    # graficando
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    plt.close(fig)
#%%  Función para calcular pérdidas
"""
La funsión discriminator_loss calcula la pérdida total del discriminador 
al evaluar su rendimiento tanto en los datos reales como en los falsos, utilizando 
la entropía cruzada binaria como métrica de pérdida. 

Esta pérdida se utiliza luego para actualizar los pesos del discriminador durante 
el entrenamiento.
"""
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # calculando pérdidas para las salidas reales. Mide qué tan bien el discriminador puede identificar los datos reales como reales.
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # calculando pérdidas para las salidas falsas. Mide qué tan bien el discriminador puede identificar los datos generados como falsos
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # obteniendo pérdida total
    total_loss = real_loss + fake_loss
    return total_loss
""" Creando función generator_loss que calcula la pérdida del generador
    Esta función mide qué tan bien el generador puede engañar al discriminador 
    haciéndole creer que los datos generados son reales.
"""
def generator_loss(fake_output):
    
    return cross_entropy(tf.ones_like(fake_output), fake_output)

""" 
La función train_step recibe:
images: Un lote de imágenes reales usado para entrenar el discriminador.
generator: El modelo generador.
discriminator: El modelo discriminador.
generator_optimizer: El optimizador usado para actualizar los parámetros del generador.
discriminator_optimizer: El optimizador usado para actualizar los parámetros del discriminador.
"""
@tf.function # convierte la función en una función de tensorflow
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    #Generando un lote de ruido aleatorio que servirá como entrada al generador para producir imágenes falsas
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # realizando cálculo de gradientes 
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generando imágenes falsas mediante el ruido de entrada (noise)
        generated_images = generator(noise, training=True)
        # usando el discriminador para evaluar las imágenes reales y falsas
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        # calculando las pérdidas del generador 
        gen_loss = generator_loss(fake_output)
        # calculando las pérdidas del discriminador 
        disc_loss = discriminator_loss(real_output, fake_output)
    # calculando los gradientes de las pérdidas con respecto a las variables entrenables del modelo generador
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # calculando los gradientes de las pérdidas con respecto a las variables entrenables del modelo discriminador
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Aplicando los gradientes para actualizar los parámetros del generador y del discriminador
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    # Retornando las pérdidas del generador y del discriminado
    return gen_loss, disc_loss
#%% Función de entrenamiento
""" Argumentos de la función
dataset: El conjunto de datos de entrenamiento, que es una colección de lotes de imágenes reales.
epochs: El número de épocas para entrenar el modelo.
generator: El modelo generador del GAN.
discriminator: El modelo discriminador del GAN.
generator_optimizer: El optimizador para el generador.
discriminator_optimizer: El optimizador para el discriminador.
"""
def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
    # generando semilla de ruido aleatorio que usará para generar y guardar imagenes durante el entrenamiento
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    for epoch in range(epochs):#Este es el bucle principal que recorre cada época de entrenamiento.
        for image_batch in dataset:#Bucle interno para los lotes de imágenes
            # para cada lote de imagenes se realiza un paso de entrenamiento y retorna las pérdidas del generador y discriminante
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)
            #Al final de cada época, se imprime la pérdida del generador y del discriminador
        print(f'Epoch {epoch + 1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')
        """
        Cada 10 épocas, se llama a la función generate_and_save_images para generar
        y guardar imágenes usando el generador y la semilla de ruido
        """
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
#%% Inicializar modelos y optimizadores
generator = make_generator_model()
discriminator = make_discriminator_model()
# creando optimizador Adam para el generador con una tasa de aprendizaje de 1e-4
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# creando optimizador Adam para el discriminador con una tasa de aprendizaje de 1e-4
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#%% Preparación del dataset
# cargando conjunto de datos MNIST
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
"""
Las imágenes originalmente están en un formato de 2D (28x28 píxeles). 
Esta línea redimensiona las imágenes a un formato de 4D, donde el último canal 
es el canal de color (en este caso, 1 porque son imágenes en escala de grises).
Y convierte las imágenes a tipo de dato float32 que es el tipo de dato utilizado para tensorflow
"""
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# normalizando los valores de los píxeles de las imágenes 
train_images = (train_images - 127.5) / 127.5  # Normalizar las imágenes a [-1, 1]
"""
Es el tamaño del buffer para mezclar los datos. En este caso, se ha configurado 
para 60.000, que es el número total de imágenes en el conjunto de datos MNIST de entrenamiento
"""
BUFFER_SIZE = 60000

BATCH_SIZE = 256 # Es el tamaño de los lotes de imágenes que se utilizarán durante el entrenamiento.
"""
creando objeto datased apartir del tensor de imagenes de entrenamiento.
Mezclando aleatoriamente los elementos del dataset utilizando un buffer de tamaño BUFFER_SIZE
y agrupando los elementos del datased en lotes de tamaño BATCH_SIZE
"""
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#%% Definiendo parámetros del ruido
# Definiendo la dimensión del vector de ruido 
noise_dim = 100 
# Definiendo el número de imágenes que se generarán para monitorear el progreso del entrenamiento
num_examples_to_generate = 16
#%% Lanzando el entrenamiento
train(train_dataset, 50, generator, discriminator, generator_optimizer, discriminator_optimizer)
