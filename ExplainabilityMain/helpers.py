import copy
import random

import numpy as np
import matplotlib.pyplot as plt
import os


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


def save_scrabble_gan_images(rows, cols, predictions, fixed_labels, name, path, show_labels=True):
    for i in range(predictions.shape[0]):
        plt.subplot(rows, cols, i+1)
        plt.imshow(predictions[i, :, :, 0].numpy().squeeze().T, cmap='gray')
        if show_labels:
            plt.title(fixed_labels[i])
        plt.axis('off')

    plt.savefig(os.path.join(path, f'{name}.png'))
    plt.close()


def save_mnist_images(predictions, images_in_single_row, path):
    rows = predictions.shape[0] // images_in_single_row
    if predictions.shape[0] % images_in_single_row != 0:
        rows += 1

    for i in range(predictions.shape[0]):
        plt.subplot(rows, images_in_single_row, i+1)
        plt.imshow(predictions[i, :, :, 0].numpy().squeeze(), cmap='gray')
        plt.axis('off')

    plt.savefig(path)
    plt.close()


def save_act_maps(org_images, locations, maps, path, rows=10, cols=10):

    for i, org_image in enumerate(org_images):
        plt.subplot(rows, cols, i*cols + 1)
        plt.imshow(org_image, cmap='gray')

        loc = copy.copy(locations)
        for j, loc in enumerate(loc):
            plt.scatter(loc[0], loc[1], s=1, c=(1, 0, 0))

        loc = copy.copy(locations)
        for j, loc in enumerate(loc):
            plt.subplot(rows, cols, i*cols + j + 2)
            plt.scatter(loc[0], loc[1], s=1, c=(1, 0, 0))
            plt.imshow(maps[i][j], cmap='gray')
            plt.axis('off')

    plt.savefig(path)
    plt.close()


def load_rand_words_from_corpus(n, word_length):
    corpus_file = os.path.join(*['Data', 'IAM', 'data', 'corpus.txt'])
    all_chars = get_all_chars_scrabble()

    with open(corpus_file, 'r') as f:
        words = [word for word in f.readline().split(' ')]
        random.shuffle(words)

    def contains(str, set):
        for s in str:
            if s not in set:
                return False
        return True

    words = [word for word in words if contains(word, set(all_chars))]

    labels = []
    for word in words:
        if len(word) == word_length:
            labels.append(word)

        if len(labels) == n:
            break
    return labels


def get_all_chars_scrabble():
    all_chars_file = os.path.join(*['Data', 'IAM', 'data', 'all_chars.txt'])
    with open(all_chars_file, 'r') as f:
        chars = [word for word in f.readline()]

        return chars


def get_labels(is_random, n_samples, type):
    if type == 'CGAN':
        if is_random:
            labels = np.random.randint(10, size=(n_samples, 1))

        else:
            labels = np.arange(10)[..., np.newaxis]

    elif type == 'ScrabbleGAN':

        if is_random:
            labels = load_rand_words_from_corpus(n_samples, 6)

        else:
            labels = ['Are', 'You', 'Mad', 'Get', 'hat', 'bat', 'vet', 'buy', 'shy', 'ray']

        chars = get_all_chars_scrabble()
        labels = np.array([[chars.index(c) for c in list(label)] for label in labels], dtype=np.int)

    else:
        labels = None

    return labels
