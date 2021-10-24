import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, \
    Input, MaxPooling2D, GlobalAveragePooling2D, Flatten, TimeDistributed, Concatenate
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import MeanSquaredError, Hinge, BinaryCrossentropy
import numpy as np

from Models.models_utils import SpatialEmbedding, sparsify, ResBlock, ResUpBlock, ResDownBlock, TripleConv, TripleDense, \
    get_activation, SelfAttentionLayer, get_labels_true_length, ctc_loss
from helpers import printProgressBar, save_scrabble_gan_images

import os
import json


def balance_gradients(gradients_of_G_d, gradients_of_G_r, alpha, v='v1'):

    if v == 'v1':
        d_grad_list = [tf.reshape(d, [-1]) for d in gradients_of_G_d]
        d_grad = Concatenate()(d_grad_list)
        d_std = tf.stop_gradient(tf.math.reduce_std(d_grad))

        r_grad_list = [tf.reshape(r, [-1]) for r in gradients_of_G_r]
        r_grad = Concatenate()(r_grad_list)
        r_std = tf.stop_gradient(tf.math.reduce_std(r_grad))

        factor = tf.divide(d_std, (1e-7+r_std))
        gradients_of_G_r_balanced = [alpha * factor * r for r in gradients_of_G_r]

    elif v == 'v2':
        stds_r, stds_d = [], []
        for d in gradients_of_G_d:
            if len(d.shape) == 0:
                stds_d.append(tf.math.reduce_std(d, keepdims=True))
            else:
                stds_d.append(tf.math.reduce_std(d, axis=0, keepdims=True))

        for r in gradients_of_G_r:
            if len(r.shape) == 0:
                stds_r.append(tf.math.reduce_std(r, keepdims=True))
            else:
                stds_r.append(tf.math.reduce_std(r, axis=0, keepdims=True))

        gradients_of_G_r_balanced = [alpha * (r_std/(1e-7+d_std)) * r for r, r_std, d_std in zip(gradients_of_G_r, stds_r, stds_d)]

    else:
        raise NotImplementedError

    gradients_of_G_balanced = [r + d for (r, d) in zip(gradients_of_G_r_balanced, gradients_of_G_d)]

    return gradients_of_G_balanced


def get_optimizers(params):
    g_lr = params['g_lr']
    G_optimizer = Adam() if g_lr is None else Adam(g_lr, beta_1=0.0, beta_2=0.999)

    d_lr = params['d_lr']
    D_optimizer = Adam() if d_lr is None else Adam(d_lr, beta_1=0.0, beta_2=0.999)

    r_lr = params['r_lr']
    R_optimizer = Adam() if g_lr is None else Adam(r_lr, beta_1=0.0, beta_2=0.999)

    return G_optimizer, D_optimizer, R_optimizer


def get_CBN_G(vocab_size, noise_dim, emb_dim, attention, attention_additive, attention_gamma_init):
    att_layer = None
    label = Input(shape=(None, ), dtype=tf.int32)
    in_z = Input(shape=(noise_dim*4, ), dtype=tf.float32)

    embedded_y = SpatialEmbedding(vocab_size=vocab_size, filter_dim=((emb_dim, noise_dim)))(label)

    shape = tf.shape(embedded_y)
    embedded_y = tf.reshape(embedded_y, (shape[0], shape[1], emb_dim, noise_dim))

    z_per_block = tf.split(in_z, 4, axis=1)
    z0, z_per_block = z_per_block[0], z_per_block[1:]

    z0_shape = tf.shape(z0)
    z0 = tf.reshape(z0, (z0_shape[0], 1, noise_dim, 1))
    z0 = tf.tile(z0, [1, tf.shape(embedded_y)[1], 1, 1])

    noisy_embedded = tf.matmul(embedded_y, z0)

    noisy_embedded = tf.squeeze(noisy_embedded, axis=3)
    shapes = [tf.shape(noisy_embedded)[k] for k in range(3)]

    x = tf.reshape(noisy_embedded, shape=(shapes[0], -1, 4, 512))

    x = ResUpBlock(resolution=(2, 2), in_ch= 512, out_ch=256, kernel_sizes=[3, 3], batch_normalization_type='conditional', upsampling_type='upsampling')([x, z_per_block[0]])
    x = ResUpBlock(resolution=(2, 2), in_ch= 256, out_ch=128, kernel_sizes=[3, 3], batch_normalization_type='conditional', upsampling_type='upsampling')([x, z_per_block[1]])
    x = ResUpBlock(resolution=(1, 2), in_ch= 128, out_ch=64, kernel_sizes=[3, 1], batch_normalization_type='conditional', upsampling_type='upsampling')([x, z_per_block[2]])

    x = BatchNormalization()(x)
    x = get_activation('relu')(x)

    if attention:
        att_layer = SelfAttentionLayer(ch=64, add_input=attention_additive, attention_gamma_init=attention_gamma_init, name='g_att_1')
        att_out = att_layer(x)
    else:
        att_out = x

    y = att_out
    y = TripleConv(channels=1, kernel_size=3, stride=(1, 1), activation='tanh')(y)

    full_model = Model(inputs=[label, in_z], outputs=y)
    att_model = Model(inputs=[label, in_z], outputs=x)
    return full_model, att_model, att_layer


def get_D(in_shape, attention, attention_additive, attention_gamma_init, last_activation):
    att_layer = None
    in_img = Input(shape=in_shape)
    x = in_img

    x = ResDownBlock(resolution=(2, 2), kernel_sizes=[3, 3], out_ch=64, weights_normalization='sn')(x)
    if attention:
        att_layer = SelfAttentionLayer(ch=64, add_input=attention_additive, attention_gamma_init=attention_gamma_init, name='d_att_1')
        att_out = att_layer(x)
    else:
        att_out = x

    y = ResDownBlock(resolution=(2, 2), kernel_sizes=[3, 3], out_ch=512, weights_normalization='sn')(att_out)
    y = ResDownBlock(resolution=(2, 2), kernel_sizes=[3, 3], out_ch=1024, weights_normalization='sn')(y)
    y = ResBlock(out_ch=1024, kernel_sizes=[3, 3], weights_normalization='sn')(y)

    y = GlobalAveragePooling2D()(y)

    y = Flatten()(y)

    y = TripleDense(1, weights_normalization='sn', normalization=None, activation=last_activation)(y)

    full_model = Model(inputs=in_img, outputs=y)
    att_model = Model(inputs=in_img, outputs=x)
    return full_model, att_model, att_layer


def get_R(args):

    if args['is_dynamic']:
        cnn_input_shape = (None, args['image_height'], 1)
    else:
        cnn_input_shape = args['image_size']

    in_img = Input(shape=cnn_input_shape)
    x = in_img

    x = TripleConv(channels=64, kernel_size=5, stride=(1, 1), normalization=None, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = TripleConv(channels=128, kernel_size=5, stride=(1, 1), normalization=None, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = TripleConv(channels=256, kernel_size=3, stride=(1, 1), normalization='bn', activation='relu')(x)

    x = TripleConv(channels=256, kernel_size=3, stride=(1, 1), normalization=None, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 2))(x)

    x = TripleConv(channels=512, kernel_size=3, stride=(1, 1), normalization='bn', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 2))(x)

    x = TripleConv(channels=512, kernel_size=3, stride=(1, 1), normalization=None, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 2))(x)

    x = TripleConv(channels=512, kernel_size=3, stride=(1, 1), normalization='bn', activation='relu')(x)

    x = tf.squeeze(x, axis=2)

    n_final = len(args['all_chars']) + 1

    x = TimeDistributed(Dense(n_final))(x)

    model = Model(inputs=in_img, outputs=x)
    return model


class ScrabbleGAN:
    def __init__(self, args):

        self.params = args
        self.emb_dim = args['emb_dim']
        self.noise_dim = args['noise_dim']
        self.all_chars = list(args['crnn_params']['all_chars'])
        self.image_height = args['crnn_params']['image_height']
        self.n_chars = len(self.all_chars)
        self.crnn_args = args['crnn_params']

        self.attention = args['attention']
        self.attention_additive = args['attention_additive']
        self.attention_gamma_init = args['attention_gamma_init']

        self.loss = args['gan_loss']
        # self.disc_last_activation = 'sigmoid' if self.loss == 'CE' else None
        self.disc_last_activation =  None

        self.bn = args['bn']

        self.G, self.G_att_model, self.G_att_layer = get_CBN_G(self.n_chars, self.noise_dim, self.emb_dim, self.attention, self.attention_additive, self.attention_gamma_init)
        self.G.summary()

        self.R = get_R(self.crnn_args)
        self.R.summary()

        self.D, self.D_att_model, self.D_att_layer = get_D((None, self.image_height, 1), self.attention, self.attention_additive, self.attention_gamma_init, self.disc_last_activation)
        self.D.summary()


    def train(self, data_loader, train_params, path):

        def get_d_loss(real_output, fake_output):

            if self.loss == 'Hinge':
                a = -tf.math.minimum(-1 + real_output, 0)
                b = -tf.math.minimum(-1 - fake_output, 0)

                real_loss = tf.reduce_mean(a)
                fake_loss = tf.reduce_mean(b)

                train_accuracy(np.ones(real_output.shape[0]), tf.sigmoid(real_output))
                train_accuracy(tf.zeros(fake_output.shape[0]), tf.sigmoid(fake_output))

            elif self.loss == 'CE':
                a = loss_func(np.ones(real_output.shape[0]), real_output)
                b = loss_func(tf.zeros(fake_output.shape[0]), fake_output)

                real_loss = tf.reduce_mean(a)
                fake_loss = tf.reduce_mean(b)

                train_accuracy(np.ones(real_output.shape[0]), tf.sigmoid(real_output))
                train_accuracy(tf.zeros(fake_output.shape[0]), tf.sigmoid(fake_output))

            elif self.loss == 'LS':
                a = loss_func(np.ones(real_output.shape[0]), real_output)
                b = loss_func(tf.zeros(fake_output.shape[0]), fake_output)

                real_loss = a
                fake_loss = b

                train_accuracy(np.ones(real_output.shape[0]), tf.sigmoid(real_output))
                train_accuracy(tf.zeros(fake_output.shape[0]), tf.sigmoid(fake_output))

            else:
                raise NotImplementedError

            total_loss = real_loss + fake_loss

            return total_loss

        def get_g_loss(fake_output):
            if self.loss == 'Hinge':
                loss = tf.reduce_mean(-fake_output)

            elif self.loss == 'CE':
                loss = loss_func(np.ones(fake_output.shape[0]), fake_output)

            elif self.loss == 'LS':
                loss = loss_func(np.ones(fake_output.shape[0]), fake_output)

            else:
                raise NotImplementedError

            return loss

        @tf.function
        def train_step(images, ints_labels, sparse_labels):
            batch_size = images.shape[0]
            noise = tf.random.normal([batch_size, self.noise_dim*4])

            with tf.GradientTape() as r_tape, tf.GradientTape() as d_tape, tf.GradientTape() as g_d_tape, tf.GradientTape() as g_r_tape:
                generated_images = self.G([ints_labels, noise], training=True)
                real_disc_output = self.D(images, training=True)
                fake_disc_output = self.D(generated_images, training=True)

                real_r_output = self.R(images, training=True)
                fake_r_output = self.R(generated_images, training=False)

                d_loss = get_d_loss(real_disc_output, fake_disc_output)
                r_loss = ctc_loss(real_r_output, sparse_labels, raw_labels_lengths)

                g_loss_from_d = get_g_loss(fake_disc_output)
                g_loss_from_r = ctc_loss(fake_r_output, sparse_labels, raw_labels_lengths)
                g_loss_from_r = train_params['gamma'] * g_loss_from_r

                g_loss = g_loss_from_r + g_loss_from_d

            gradients_of_D = d_tape.gradient(d_loss, self.D.trainable_variables)
            D_optimizer.apply_gradients(zip(gradients_of_D, self.D.trainable_variables))

            gradients_of_R = r_tape.gradient(r_loss, self.R.trainable_variables)
            R_optimizer.apply_gradients(zip(gradients_of_R, self.R.trainable_variables))

            # if epoch % train_params['critic_iterations_per_generator'] == 0:
            if train_params['gradient_balancing']:

                gradients_of_G_d = g_d_tape.gradient(g_loss_from_d, self.G.trainable_variables)
                gradients_of_G_r = g_r_tape.gradient(g_loss_from_r, self.G.trainable_variables)
                gradients_of_G_balanced = balance_gradients(gradients_of_G_d, gradients_of_G_r, train_params['alpha'])

            else:
                gradients_of_G_balanced = g_d_tape.gradient(g_loss, self.G.trainable_variables)

            G_optimizer.apply_gradients(zip(gradients_of_G_balanced, self.G.trainable_variables))

            d_loss_mat(d_loss)
            g_loss_mat(g_loss)
            r_loss_mat(r_loss)

            return g_loss, g_loss_from_d, g_loss_from_r, d_loss, r_loss

        if self.loss == 'CE':
            loss_func = BinaryCrossentropy()
        elif self.loss == 'LS':
            loss_func = MeanSquaredError()


        train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        d_loss_mat = tf.keras.metrics.Mean(name='d_loss')
        g_loss_mat = tf.keras.metrics.Mean(name='g_loss')
        r_loss_mat = tf.keras.metrics.Mean(name='r_loss')

        G_optimizer, D_optimizer, R_optimizer = get_optimizers(train_params)

        fixed_labels_arrays = [['are', 'You', 'mad', 'sad', 'but', 'bit', 'Tim', 'Tam'],
                               ['hero', 'love', 'need', 'guts', 'time', 'luck', 'must', 'wood'],
                               ['Hello', 'Heavy', 'Sight', 'jolly', 'pasta', 'toast', 'Smile', 'Cross'],
                               ['Knight', 'Castle', 'Bishop', 'Cheese', 'Fisher', 'Morphy', 'Parrot', 'carrot'],
                               ['Airline', 'alcohol', 'holiday', 'libarty', 'Nuclear', 'Protein', 'Tension', 'display']]

        fixed_noise_arrays = [tf.random.normal([len(fixed_labels_arrays[0]), self.noise_dim*4]) for _ in range(len(fixed_labels_arrays))]
        fixed_ints_labels_arrays = [np.array([[self.all_chars.index(c) for c in list(label)] for label in fixed_labels], dtype=np.int) for fixed_labels in fixed_labels_arrays]

        for epoch in range(train_params['epochs']):
            d_loss_mat.reset_states(), g_loss_mat.reset_states(), r_loss_mat.reset_states(), train_accuracy.reset_states()

            batches = data_loader.get_batches()
            n_batches = data_loader.get_num_iteration()
            for i, (batch_images, text_labels) in enumerate(batches):
                sparse_labels = sparsify(text_labels, self.all_chars)
                raw_labels_lengths = get_labels_true_length(text_labels)

                ints_labels = np.array([[self.all_chars.index(c) for c in list(label)] for label in text_labels], dtype=np.int)
                g_loss, g_d_loss, g_r_loss,  d_loss, r_loss = train_step(batch_images, ints_labels, sparse_labels)

                if i + 1 == n_batches:
                    printProgressBar(i+1, n_batches, prefix = f'Epoch {epoch+1}, batch {i+1}/{n_batches}:',
                                     suffix = f'Complete, batch loss: D_acc: {"%.3f" %train_accuracy.result()}, G: {"%.3f" %g_loss_mat.result()}, R: {"%.3f" %r_loss_mat.result()}', length = 50)
                else:
                    printProgressBar(i+1, n_batches, prefix = f'Epoch {epoch+1}, batch {i+1}/{n_batches}:',
                                     suffix = f'Complete, batch loss: D: {"%.3f" %d_loss}, G_from_d: {"%.3f" %g_d_loss}, G_from_r:,{"%.3f" %g_r_loss} R: {"%.3f" %r_loss}', length = 50)

            for j, (fixed_noise, fixed_labels, fixed_ints_labels) in enumerate(zip(fixed_noise_arrays, fixed_labels_arrays, fixed_ints_labels_arrays)):
                test_predictions = self.G([fixed_ints_labels, fixed_noise])
                save_scrabble_gan_images(2, 4, test_predictions, fixed_labels, f'{epoch}_images_{j}', os.path.join(path, 'images'))

            l = fixed_labels_arrays[3][0]
            self.generate_single(l,  f'{epoch}_shared_images', os.path.join(path, 'images'))

            self.save_model(path)
            for l in self.G.layers:
                if l.name == 'g_att_1' and self.attention_additive:
                    print(l.name, l.weights[0])
            for l in self.D.layers:
                if l.name == 'd_att_1' and self.attention_additive:
                    print(l.name, l.weights[0])

    def save_model(self, path):

        weights_path = os.path.join(path, 'weights')

        self.G.save_weights(os.path.join(weights_path, 'G'))
        self.D.save_weights(os.path.join(weights_path, 'D'))
        self.R.save_weights(os.path.join(weights_path, 'R'))

        json.dump(self.params, open(os.path.join(path, 'params.json'), 'w'))

    def load_model(self, path):

        weights_path = os.path.join(path, 'weights')
        self.G.load_weights(os.path.join(weights_path, 'G'))
        self.D.load_weights(os.path.join(weights_path, 'D'))
        self.R.load_weights(os.path.join(weights_path, 'R'))

    def generate(self, labels_arrays, noise_arrays, path):
        """

        :param labels_arrays: [[length 4], [length 7], [...]]
        :param noise_arrays:
        :param path:
        :return:
        """
        ints_labels_arrays = [np.array([[self.all_chars.index(c) for c in list(label)] for label in fixed_labels], dtype=np.int) for fixed_labels in labels_arrays]

        for j, (fixed_labels, fixed_noise, fixed_ints_labels) in enumerate(zip(labels_arrays, noise_arrays, ints_labels_arrays)):
            test_predictions = self.G([fixed_ints_labels, fixed_noise])
            save_scrabble_gan_images(2, 4, test_predictions, fixed_labels, f'inference_images_{j}.png', path)

    def generate_single(self, label, name, path):

        ints_label_array = np.array([[self.all_chars.index(c) for c in list(label)] for _ in range(8)], dtype=np.int32)
        noises = np.random.normal(size=(8, self.noise_dim*4))

        test_predictions = self.G([ints_label_array, noises])
        save_scrabble_gan_images(2, 4, test_predictions, label, name, path, show_labels=False)
