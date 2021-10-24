import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Input, Flatten, Reshape, Concatenate, Lambda, LeakyReLU

import tensorflow_probability as tfp
tfd = tfp.distributions
import os
from helpers import printProgressBar
from Models.models_utils import TripleDense, TripleConv, ResBlock, ResUpBlock, get_activation, TripleTConv
import json


def save_mnist_images(predictions, epoch, name, path):
    n = int(np.sqrt(predictions.shape[0]))
    for i in range(predictions.shape[0]):
        plt.subplot(n, n, i+1)
        plt.imshow(predictions[i, :, :, 0].numpy().squeeze(), cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(path, f"epoch_{epoch}_image_{name}.png"))
    plt.close()


def get_optimizers(params):
    g_lr = params['g_lr']
    if g_lr is None:
        generator_optimizer = Adam()
    else:
        generator_optimizer = Adam(g_lr, beta_1=0.5)
    d_lr = params['d_lr']
    if d_lr is None:
        discriminator_optimizer = Adam()
    else:
        discriminator_optimizer = Adam(d_lr, beta_1=0.5)
    q_lr = params['q_lr']
    if q_lr is None:
        q_optimizer = Adam()
    else:
        q_optimizer = Adam(q_lr, beta_1=0.5)

    return generator_optimizer, discriminator_optimizer, q_optimizer


def get_discriminator(n_classes, n_codes):

    in_img = Input(shape=(28, 28, 1))
    x = in_img
    x = TripleConv(channels=64, kernel_size=4, stride=(2, 2), weights_normalization=None, normalization='bn', activation=None)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = TripleConv(channels=128, kernel_size=4, stride=(2, 2), weights_normalization=None, normalization='bn', activation=None)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)

    x = TripleDense(size=1024, weights_normalization=None, normalization='bn', activation=None)(x)
    x = LeakyReLU(alpha=0.1)(x)
    d_out = TripleDense(size=1, weights_normalization=None, normalization=None, activation='sigmoid')(x)
    d_out = tf.cast(d_out, dtype=tf.float64)

    d_model = Model(inputs=in_img, outputs=d_out)

    mutual = TripleDense(size=128, weights_normalization=None, normalization='bn', activation=None)(x)
    mutual = LeakyReLU(alpha=0.1)(mutual)

    cat_code = TripleDense(size=n_classes, weights_normalization=None, normalization=None, activation='softmax')(mutual)

    out_codes = []

    for i in range(n_codes):
        est_mean = TripleDense(size=1, weights_normalization=None, normalization=None, activation=None)(mutual)

        est_var = TripleDense(size=1, weights_normalization=None, normalization=None, activation=None)(mutual)
        est_var = Lambda(lambda x: tf.math.exp(x))(est_var)

        out_codes.append(est_mean)
        out_codes.append(est_var)

    enc_model = Model(inputs=in_img, outputs=[d_out, cat_code] + out_codes)

    return d_model, enc_model


def get_generator(latent_dim, n_classes, n_codes):

    in_noise = Input(shape=(latent_dim, ))
    in_categorical_code = Input(shape=(1, ), dtype=tf.int32)
    in_continuous_code = Input(shape=(n_codes, ))

    in_cat_emb = tf.one_hot(depth=n_classes, indices=in_categorical_code)
    cat_ebd_shape = in_cat_emb.get_shape()
    in_cat_emb = Reshape((cat_ebd_shape[2], ))(in_cat_emb)

    x = Concatenate()([in_noise, in_cat_emb, in_continuous_code])
    x = Flatten()(x)

    x = TripleDense(size=1024, weights_normalization=None, normalization='bn', activation='relu')(x)
    x = TripleDense(size=7*7*128, weights_normalization=None, normalization='bn', activation='relu')(x)
    x = Reshape((7, 7, 128))(x)

    x = TripleTConv(channels=128, kernel_size=4, stride=(2, 2), weights_normalization=None, normalization='bn', activation='relu')(x)
    x = TripleTConv(channels=64, kernel_size=4, stride=(2, 2), weights_normalization=None, normalization='bn', activation='relu')(x)
    x = TripleTConv(channels=1, kernel_size=4, stride=(1, 1), weights_normalization=None, normalization=None, activation='tanh')(x)

    model = Model(inputs=[in_noise, in_categorical_code, in_continuous_code], outputs=x)

    return model


class infoGAN:
    def __init__(self, params):

        self.latent_dim = params['latent_dim']
        self.n_classes = params['n_classes']

        self.params = params
        self.n_continuous_codes = params['n_continuous_codes']

        self.discriminator, self.Qnet = get_discriminator(self.n_classes, self.n_continuous_codes)
        self.generator = get_generator(self.latent_dim, self.n_classes, self.n_continuous_codes)

        self.discriminator.summary()
        self.generator.summary()
        self.Qnet.summary()

    def train(self, data_loader, path, params):
        # inner functions

        def generator_loss(fake_output):

            loss = binary_entropy_loss(tf.ones(fake_output.shape[0], dtype=tf.float64), tf.squeeze(fake_output))
            loss = tf.cast(loss, dtype=tf.float64)
            return loss

        def encoding_loss(est_c, cat_codes, means_and_variances, continuous_codes):
            cat_code_loss = cross_entropy_loss(cat_codes, est_c)

            log_loss = 0
            for i in range(self.n_continuous_codes):
                mu = Flatten()(means_and_variances[2*i])

                sigma = Flatten()(means_and_variances[2*i+1])

                x = Flatten()(continuous_codes[:, i])

                c_dist = tfd.Normal(loc=mu, scale=sigma)
                log_prob = c_dist.log_prob(x)
                log_loss += tf.reduce_mean(tf.boolean_mask(-log_prob, tf.math.is_finite(-log_prob)))

            log_loss = tf.cast(log_loss, dtype=tf.float64)
            cat_code_loss = tf.cast(cat_code_loss, dtype=tf.float64)
            return cat_code_loss, log_loss

        @tf.function
        def train_step(image_batch):
            batch_size = image_batch.shape[0]
            noise = tf.random.normal([batch_size, self.latent_dim])
            true_category_code = np.random.randint(0, self.n_classes, size=(batch_size, 1))
            true_continuous_codes = np.random.uniform(-1, 1, (batch_size, self.n_continuous_codes))

            with tf.GradientTape() as d_tape:
                d_out_real = self.discriminator(image_batch, training=True)
                d_loss_real = binary_entropy_loss(tf.ones(d_out_real.shape[0]), tf.squeeze(d_out_real))
                train_loss(d_loss_real)
                train_d_accuracy(np.ones(d_out_real.shape[0]), tf.squeeze(d_out_real))

            with tf.GradientTape() as g_tape, tf.GradientTape() as q_tape:
                generated_images = self.generator([noise, true_category_code, true_continuous_codes], training=True)

                q_outs = self.Qnet(generated_images, training=True)
                d_out_fake, est_cat, est_means_vars = q_outs[0], q_outs[1], q_outs[2:]

                gen_loss = generator_loss(d_out_fake)

                d_loss_fake = binary_entropy_loss(tf.zeros(d_out_fake.shape[0]), tf.squeeze(d_out_fake))
                train_loss(d_loss_fake)

                cat_code_loss, c_loss = encoding_loss(est_cat, true_category_code, est_means_vars, true_continuous_codes)
                enc_alpha = 1

                enc_loss = enc_alpha*(100*cat_code_loss + 0.1*c_loss)
                q_loss = d_loss_fake + enc_loss

                g_loss = gen_loss + enc_loss

            d_gradients = d_tape.gradient(d_loss_real, self.discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            g_optimzer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

            q_gradients = q_tape.gradient(q_loss, self.Qnet.trainable_variables)
            q_optimizer.apply_gradients(zip(q_gradients, self.Qnet.trainable_variables))

            train_loss(gen_loss)
            train_loss(q_loss)

            train_d_accuracy(np.ones(d_out_fake.shape[0]), tf.squeeze(d_out_fake))
            train_q_accuracy(true_category_code, est_cat)
            train_q_loss(q_loss)

            return d_loss_real+d_loss_fake, gen_loss, q_loss
            # End inner function

        fixed_seed = tf.random.normal([100, self.latent_dim])
        fixed_cat = np.repeat(np.arange(self.n_classes), 10, axis=0).reshape(-1, 1)
        fixed_c_range = np.repeat(np.linspace(start=-1, stop=1, num=10).reshape((1, 10)), 10, axis=0).flatten()
        fixed_c_random = np.random.uniform(-1, 1, size=(100, self.n_continuous_codes))

        ## measurments
        train_d_accuracy = tf.keras.metrics.BinaryAccuracy()
        train_q_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # losses
        train_loss = tf.keras.metrics.Mean()
        train_q_loss = tf.keras.metrics.Mean()
        cross_entropy_loss = SparseCategoricalCrossentropy()
        binary_entropy_loss = tf.keras.losses.BinaryCrossentropy()

        g_optimzer, d_optimizer, q_optimizer = get_optimizers(params)

        for epoch in range(params['epochs']):
            train_loss.reset_states()
            train_d_accuracy.reset_states()
            train_q_accuracy.reset_states()
            train_q_loss.reset_states()

            batches = data_loader.get_batches_iterator(size = params['batch_size'])
            n_batches = data_loader.get_n_batches(size = params['batch_size'])

            for i, (image_batch, _) in enumerate(batches):
                disc_loss, gen_loss, q_loss = train_step(image_batch)

                if i + 1 == n_batches:
                    printProgressBar(i+1, n_batches, prefix = f'Epoch {epoch+1}, batch {i+1}/{n_batches}:',
                                     suffix = f'Complete, loss: {"%.3f" %train_loss.result()}, discriminator accuracy: {"%.3f" %train_d_accuracy.result()}, '
                                              f'codes accuracy: {"%.3f" %train_q_accuracy.result()}, continuous loss: {"%.3f" %train_q_loss.result()}', length = 50)
                else:
                    printProgressBar(i+1, n_batches, prefix = f'Epoch {epoch+1}, batch {i+1}/{n_batches}:',
                                     suffix = f'Complete, generator_loss: {"%.3f" %gen_loss}, discriminator loss: {"%.3f" % disc_loss}, q_loss: {"%.3f" %q_loss}', length = 50)

            data_loader.init_iterator()

            predictions_1 = self.generator([fixed_seed, fixed_cat, fixed_c_random], training=False)
            save_mnist_images(predictions_1, epoch+1, 'random_seeds', os.path.join(path, 'images'))

            for i in range(3):
                fixed_seed_i_tiled = np.array([fixed_seed[i, :]] * 100)
                for j in range(self.n_continuous_codes):
                    fixed_c_j = np.zeros((100, self.n_continuous_codes))
                    fixed_c_j[:, j] = fixed_c_range
                    predictions = self.generator([fixed_seed_i_tiled, fixed_cat, fixed_c_j], training=False)

                    save_mnist_images(predictions, epoch+1, f'same_z_{i}_code_{j}', os.path.join(path, 'images'))

            self.save_model(path)

    def save_model(self, path):
        json.dump(self.params, open(os.path.join(path, 'params.json'), 'w'))
        weights_path = os.path.join(path, 'weights')
        self.Qnet.save_weights(os.path.join(weights_path, 'Qnet'))
        self.generator.save_weights(os.path.join(weights_path, 'gen'))
        self.discriminator.save_weights(os.path.join(weights_path, 'disc'))

    def load_model(self, path):
        weights_path = os.path.join(path, 'weights')
        self.generator.load_weights(os.path.join(weights_path, 'gen'))
        self.discriminator.load_weights(os.path.join(weights_path, 'disc'))

