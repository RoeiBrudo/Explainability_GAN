import os
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Flatten, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, Hinge, MeanSquaredError
from tensorflow.keras.metrics import BinaryAccuracy, Mean

from helpers import printProgressBar, save_mnist_images
from Models.models_utils import TripleDense, TripleConv, ResUpBlock, get_activation, SelfAttentionLayer


def get_optimizers(params):
    g_lr = params['g_lr']
    G_optimizer = Adam() if g_lr is None else Adam(g_lr)

    d_lr = params['d_lr']
    D_optimizer = Adam() if d_lr is None else Adam(d_lr)

    return G_optimizer, D_optimizer


def get_discriminator(attention, attention_additive, attention_gamma_init, last_activation):
    att_layer = None
    in_img = Input(shape=(28, 28, 1))

    x = in_img
    x = TripleConv(channels=64, kernel_size=3, stride=(1, 1), weights_normalization='sn', normalization=None, activation='l_relu')(x)

    if attention:
        att_layer = SelfAttentionLayer(ch=64, add_input=attention_additive, attention_gamma_init=attention_gamma_init, name='d_att_1')
        att_out = att_layer(x)
    else:
        att_out = x

    y = att_out
    y = TripleConv(channels=128, kernel_size=3, stride=(2, 2), weights_normalization='sn', normalization='bn', activation='l_relu')(y)
    y = TripleConv(channels=256, kernel_size=3, stride=(2, 2), weights_normalization='sn', normalization='bn', activation='l_relu')(y)

    y = Flatten()(y)

    y = TripleDense(size=512, weights_normalization='sn', normalization='sn', activation='l_relu')(y)

    y = TripleDense(size=1, weights_normalization='sn', normalization=None, activation=last_activation)(y)

    full_model = Model(inputs=in_img, outputs=y)
    att_model = Model(inputs=in_img, outputs=x)
    return full_model, att_model, att_layer


def get_generator(latent_dim, attention, attention_additive, attention_gamma_init):
    att_layer = None
    in_noise = Input(shape=(latent_dim, ))

    x = in_noise

    x = TripleDense(size=7*7*128, weights_normalization=None, normalization=None, activation='relu')(x)
    x = Reshape((7, 7, 128))(x)

    x = ResUpBlock(resolution=(2, 2), kernel_sizes=[3, 3], in_ch = 128, out_ch=64, weights_normalization='sn', batch_normalization_type='regular', upsampling_type='upsampling')(x)
    x = ResUpBlock(resolution=(2, 2), kernel_sizes=[3, 3], in_ch = 64, out_ch=32, weights_normalization='sn', batch_normalization_type='regular', upsampling_type='upsampling')(x)
    x = BatchNormalization()(x)
    x = get_activation('relu')(x)

    if attention:
        att_layer = SelfAttentionLayer(32, add_input=attention_additive, attention_gamma_init=attention_gamma_init, name='g_att_1')
        att_out = att_layer(x)
    else:
        att_out = x

    y = att_out

    y = TripleConv(channels=1, kernel_size=3, stride=(1, 1), weights_normalization=None, normalization=None, activation='tanh')(y)

    full_model = Model(inputs=in_noise, outputs=y)
    att_model = Model(inputs=in_noise, outputs=x)

    return full_model, att_model, att_layer


class MNISTGAN:
    def __init__(self, params):

        self.latent_dim = params['latent_dim']
        self.attention = params['attention']
        self.attention_additive = params['attention_additive']
        self.attention_gamma_init = params['attention_gamma_init']
        self.loss = params['gan_loss']
        self.params = params

        self.disc_last_activation = 'sigmoid' if self.loss == 'CE' else None


        self.G, self.G_att_model, self.G_att_layer = get_generator(self.latent_dim, self.attention, self.attention_additive, self.attention_gamma_init)
        self.D, self.D_att_model, self.D_att_layer = get_discriminator(self.attention, self.attention_additive, self.attention_gamma_init, self.disc_last_activation)

        self.D.summary()
        self.G.summary()

    def train(self, data_loader, path, params):
        # inner functions

        def discriminator_loss(real_output, fake_output):
            real_loss = gan_loss(tf.ones(real_output.shape[0]), real_output)
            fake_loss = gan_loss(tf.zeros(fake_output.shape[0]), fake_output)

            train_accuracy(np.ones(real_output.shape[0]), real_output)
            train_accuracy(np.ones(fake_output.shape[0]), fake_output)

            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(fake_output):
            return gan_loss(tf.ones(fake_output.shape[0]), fake_output)

        @tf.function
        def train_step(image_batch):
            batch_size = image_batch.shape[0]
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.G(noise, training=True)
                real_output = self.D(image_batch, training=True)
                fake_output = self.D(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            train_loss(disc_loss)
            train_loss(gen_loss)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
            G_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))

            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)
            D_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))

            return disc_loss, gen_loss
            # End inner function

        if self.loss == 'CE':
            gan_loss = BinaryCrossentropy()
        elif self.loss == 'Hinge':
            gan_loss = Hinge()
        elif self.loss == 'LS':
            gan_loss = MeanSquaredError()
        else:
            raise NotImplementedError

        G_optimizer, D_optimizer = get_optimizers(params)

        train_loss = Mean(name='train_loss')
        train_accuracy = BinaryAccuracy(name='train_accuracy')

        fixed_seed = tf.random.normal([100, self.latent_dim])

        for epoch in range(params['epochs']):
            train_loss.reset_states(), train_accuracy.reset_states()

            batches = data_loader.get_batches_iterator(size = params['batch_size'])
            n_batches = data_loader.get_n_batches(size = params['batch_size'])

            for i, (image_batch, _) in enumerate(batches):
                disc_loss, gen_loss = train_step(image_batch)

                if i + 1 == n_batches:
                    printProgressBar(i+1, n_batches, prefix = f'Epoch {epoch+1}, batch {i+1}/{n_batches}:',
                                     suffix = f'Complete, loss: {train_loss.result()}, discriminator accuracy: {train_accuracy.result()}', length = 50)
                else:
                    printProgressBar(i+1, n_batches, prefix = f'Epoch {epoch+1}, batch {i+1}/{n_batches}:',
                                     suffix = f'Complete, generator_loss: {gen_loss}, discriminator loss: {disc_loss}', length = 50)

            data_loader.init_iterator()

            if self.attention and self.attention_additive:
                for l in self.G.layers:
                    if l.name == 'g_att_1':
                        print(l.name, l.weights[0])

                for l in self.D.layers:
                    if l.name == 'd_att_1':
                        print(l.name, l.weights[0])

            predictions = self.G(fixed_seed, training=False)
            save_mnist_images(predictions, 10, os.path.join(os.path.join(path, 'images'), f'epoch_{epoch + 1}.png'))

            self.save_model(path)

    def generate(self, noise, path):
        predictions = self.G(noise, training=False)
        save_mnist_images(predictions, 10, path)

    def save_model(self, path):
        json.dump(self.params, open(os.path.join(path, 'params.json'), 'w'))

        weights_path = os.path.join(path, 'weights')
        self.G.save_weights(os.path.join(weights_path, 'gen'))
        self.D.save_weights(os.path.join(weights_path, 'disc'))

    def load_model(self, path):
        weights_path = os.path.join(path, 'weights')
        self.G.load_weights(os.path.join(weights_path, 'gen')).expect_partial()
        self.D.load_weights(os.path.join(weights_path, 'disc')).expect_partial()

