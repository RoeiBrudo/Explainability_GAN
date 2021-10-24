import random
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, \
    Add, MaxPooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.constraints import UnitNorm

from models_utils import ResDownBlock
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import os


fixed_labels_txt = ['Shalom', 'belief', 'Castle', 'Knight', 'Bishop', 'French', 'Hebrew', 'legacy', 'Summer', 'Morphy']

all_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

fixed_labels = np.array([[all_chars.index(c) for c in chars]for chars in fixed_labels_txt])

random_words_by_length = {}

with open('random_words.txt', 'r') as random_words_file:
    lines = random_words_file.readlines()
    all_random_words = [line.rstrip() for line in lines]
    for word in all_random_words:
        l = len(word)
        if l in random_words_by_length.keys():
            random_words_by_length[l].append(word)
        else:
            random_words_by_length[l] = [word]


def get_random_labels(k):
    random_length = np.random.randint(1, 10)
    labels_texts = random.choices(random_words_by_length[random_length], k=k)
    labels = np.array([[all_chars.index(c) for c in chars]for chars in labels_texts])

    print(labels.shape)
    return labels

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '!', printEnd = ""):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()



def plot_images(org_images, shifted, step, eps, path, transpose=False):

    k = len(org_images)
    paths = [os.path.join(path, f'direction_{i}') for i in range(k)]

    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    single_plt_rows, cols = len(org_images[0]), len(shifted[0][0])+1

    for k_i in range(k):

        printProgressBar(k_i+1, k, prefix = 'Plotting epoch results', length = 50)

        plt.figure(0)
        for i in range(single_plt_rows):

            ax = plt.subplot2grid((single_plt_rows, cols), (i, 0))
            if transpose:
                ax.imshow(org_images[k_i][i, :, :, 0].T, cmap='gray')
            else:
                ax.imshow(org_images[k_i][i, :, :, 0], cmap='gray')
            if i == 0:
                ax.set_title('org-image', fontdict ={'fontsize': 5}),
            ax.axis('off')

            for j in range(1, cols):
                ax = plt.subplot2grid((single_plt_rows, cols), (i, j))
                if transpose:
                    ax.imshow(shifted[k_i][i][j-1, :, :, 0].T, cmap='gray')
                else:
                    ax.imshow(shifted[k_i][i][j-1, :, :, 0], cmap='gray')

                if i == 0:
                    ax.set_title(f'{eps[j-1]:10.1f}', fontdict ={'fontsize': 5}),
                ax.axis('off')

        plt.savefig(os.path.join(path, os.path.join(f'direction_{k_i}', f'step_{step}.png')))
        plt.close()


def generate_random_inputs(k, d, batch_size):

    noise = tf.random.normal([batch_size, d])

    ks = np.random.randint(0, k, size=batch_size, dtype=int)

    eps = np.random.uniform(-6, 6, size=batch_size)
    eps = np.sign(eps) * np.maximum(np.abs(eps), 0.5)

    return noise, ks, eps


def get_R_model(k, type):

    img_org = Input(shape=(32, None, 1))
    img_shifted = Input(shape=(32, None, 1))
    images = Concatenate()([img_org, img_shifted])

    images = ResDownBlock(resolution=(2, 2), kernel_sizes=[3, 3], out_ch=256)(images)
    images = ResDownBlock(resolution=(2, 2), kernel_sizes=[3, 3], out_ch=512)(images)
    images = ResDownBlock(resolution=(2, 2), kernel_sizes=[3, 3], out_ch=1024)(images)


    k_layer = Conv2D(256, 3, 2, padding='same')(images)
    k_layer = BatchNormalization()(k_layer)
    k_layer = ReLU()(k_layer)

    k_layer = Conv2D(512, 3, 2, padding='same')(k_layer)
    k_layer = BatchNormalization()(k_layer)
    k_layer = ReLU()(k_layer)

    if type == 'ScrabbleGAN':
        k_layer = GlobalAveragePooling2D()(k_layer)

    k_layer = Flatten()(k_layer)

    k_layer = Dense(250, activation='relu')(k_layer)
    k_layer = Dense(k, activation='softmax')(k_layer)

    ###

    eps_layer = Conv2D(256, 3, 2, padding='same')(images)
    eps_layer = BatchNormalization()(eps_layer)
    eps_layer = ReLU()(eps_layer)

    eps_layer = Conv2D(512, 3, 2, padding='same')(eps_layer)
    eps_layer = BatchNormalization()(eps_layer)
    eps_layer = ReLU()(eps_layer)

    if type == 'ScrabbleGAN':
        eps_layer = GlobalAveragePooling2D()(eps_layer)

    eps_layer = Flatten()(eps_layer)

    eps_layer = Dense(250)(eps_layer)
    eps_layer = Dense(1)(eps_layer)

    model = Model(inputs=[img_org, img_shifted], outputs=[k_layer, eps_layer], name='R_model')

    return model


def get_A_model(d, k):
    k = Input(shape=(k, ))
    org_noise = Input(shape=(d, ))

    z_noisy = Dense(d, use_bias=True, kernel_constraint=UnitNorm(axis=1), name='A_matrix')(k)
    z_noisy = Add()([z_noisy, org_noise])

    model = Model(inputs=[k, org_noise], outputs=z_noisy, name="A_model")
    return model


class GanLatentDiscover:
    def __init__(self, G_model, gen_type, k):
        self.gen_type = gen_type
        self.G_model = G_model

        self.d = 128

        self.k = k

        self.R_model = get_R_model(self.k, gen_type)
        self.A_model = get_A_model(self.d, self.k)

        self.A_model.summary()
        self.R_model.summary()

    def run_g_forward(self, seeds, labels):
        images_org = self.G_model([seeds, labels]).numpy()

        return images_org

    def get_images_for_plots(self, seeds, labels, direction_moves, fixed_eps):

        org_images = [] ## [ 10 images, 10 images, ..., k_times]
        shifted = [] ## [[all eps image1 array, all eps images array2 ... 10 images]  ], [[all eps images array1, all eps images array2 ... 10 times]  , ... k times]

        for k_i in range(self.k):

            images_org = self.run_g_forward(seeds[k_i], labels)
            org_images.append(images_org)

            shifted_images_ki = []
            for j in range(10): ## num images
                fixed_ks_i_j = np.zeros((direction_moves, self.k))
                fixed_ks_i_j[:, k_i] = fixed_eps
                fixed_seed_ki_j = np.array([seeds[k_i][j, :], ]*direction_moves)

                shifted_fixed_noises_ki_j = self.A_model([fixed_ks_i_j, fixed_seed_ki_j])

                if labels is None:
                    l = None
                else:
                    l = np.array([labels[j]]*direction_moves)
                shifted_images_ki_j = self.run_g_forward(shifted_fixed_noises_ki_j, l)
                shifted_images_ki.append(shifted_images_ki_j)

            shifted.append(shifted_images_ki)

        return org_images, shifted

    def train(self, path, epochs, steps_per_epoch, batch_size):
        @tf.function
        def train_c_step(noise, labels, ks, eps, k_vectors):
            with tf.GradientTape() as r_tape, tf.GradientTape() as a_tape:

                shifted_noises = self.A_model([k_vectors, noise])
                img_org, img_shifted = self.G_model([noise, labels]), self.G_model([shifted_noises, labels])

                est_k, est_eps = self.R_model([img_org, img_shifted])

                k_loss_step = cross_entropy_loss(ks, est_k)
                eps_error_step = mae_loss(eps, est_eps)

                loss = k_loss_step + 0.25*eps_error_step
                r_gradients = r_tape.gradient(loss, self.R_model.trainable_variables)
                r_optimizer.apply_gradients(zip(r_gradients, self.R_model.trainable_variables))
                a_gradients = a_tape.gradient(loss, self.A_model.trainable_variables)
                a_optimizer.apply_gradients(zip(a_gradients, self.A_model.trainable_variables))

            k_train_accuracy(ks, est_k)
            eps_train_error(eps_error_step)
            return k_loss_step, eps_error_step

        cross_entropy_loss = SparseCategoricalCrossentropy()
        mae_loss = MeanAbsoluteError()
        r_optimizer = Adam(1e-5)
        a_optimizer = Adam(1e-5)

        eps_train_error = tf.keras.metrics.Mean(name='train_loss')
        eps_test_error = tf.keras.metrics.Mean(name='test_loss')

        k_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        k_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        test_steps = 10
        direction_moves = 10
        fixed_seeds = [tf.random.normal([10, self.d]) for _ in range(self.k)]
        fixed_eps = np.linspace(-6, 6, direction_moves)
        fixed_eps = np.sign(fixed_eps) * np.maximum(np.abs(fixed_eps), np.zeros(direction_moves)+0.5)


        for epoch in range(epochs):

            k_test_accuracy.reset_states()
            eps_test_error.reset_states()
            k_train_accuracy.reset_states()
            eps_train_error.reset_states()

            for step in range(steps_per_epoch):
                train_noise, train_ks, train_eps = generate_random_inputs(self.k, self.d, batch_size)
                train_k_vectors = np.zeros(shape=(batch_size, self.k))
                train_k_vectors[np.arange(batch_size), train_ks] = train_eps

                labels = get_random_labels(k=batch_size)

                k_loss_step, eps_error_step = train_c_step(train_noise, labels, train_ks, train_eps, train_k_vectors)

                if step + 1 == steps_per_epoch:
                    printProgressBar(step+1, steps_per_epoch, prefix = f'Epoch {epoch+1}',
                                     suffix = f'Complete, k accuracy: {k_train_accuracy.result()} eps loss: {eps_train_error.result()}', length = 50)
                else:
                    printProgressBar(step+1, steps_per_epoch, prefix = f'Epoch {epoch+1}, batch {step+1}/{steps_per_epoch}:',
                                     suffix = f'Complete, eps loss: {eps_error_step}, k loss: {k_loss_step}', length = 50)

            for test_step in range(test_steps):
                test_noise, test_ks, test_eps = generate_random_inputs(self.k, self.d, batch_size)
                test_k_vectors = np.zeros(shape=(batch_size, self.k))
                test_k_vectors[np.arange(batch_size), test_ks] = test_eps

                shifted_noises = self.A_model([test_k_vectors, test_noise])

                img_org = self.run_g_forward(test_noise, labels),
                img_shifted = self.run_g_forward(shifted_noises, labels)

                test_est_k, test_est_eps = self.R_model([img_org, img_shifted])

                k_test_accuracy(test_ks, test_est_k)
                eps_test_error(mae_loss(test_eps, test_est_eps))

                printProgressBar(test_step+1, test_steps, prefix = f'Epoch {epoch+1}, test_step {test_step+1}/{test_steps}:',
                                 suffix= f'k_acc: {k_test_accuracy.result()}, eps_error: {eps_test_error.result()}', length = 50)

            # if epoch > 50 and (epoch % 10 == 0):
            org_images, shifted = self.get_images_for_plots(fixed_seeds, fixed_labels, direction_moves, fixed_eps)
            plot_images(org_images, shifted, epoch, fixed_eps, os.path.join(path, 'images'), transpose=False)

    def save_model(self, path):
        weights_path = os.path.join(path, 'weights')
        self.A_model.save_weights(os.path.join(weights_path, 'A'))
        self.R_model.save_weights(os.path.join(weights_path, 'R'))

    def load_model(self, path):

        weights_path = os.path.join(path, 'weights')
        self.A_model.load_weights(os.path.join(weights_path, 'A'))
        self.R_model.load_weights(os.path.join(weights_path, 'R'))
