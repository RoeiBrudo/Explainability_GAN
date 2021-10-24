import numpy as np
import argparse
import os
import json

from helpers import get_labels, save_act_maps


def train_GAN_MNIST(model_out_path):
    from Models.GAN import MNISTGAN
    from Data.MNIST.mnist_data_loader import MNIST_LOADER

    data_loader = MNIST_LOADER()
    print("Data Done")

    GAN_PARAMS ={
        'type': 'GAN',
        'latent_dim': 150,

        'attention': True,
        'attention_additive': True,
        'attention_gamma_init': 1.0,

        'gan_loss': 'Hinge',   # Hinge, LS
    }

    paths = [os.path.join(model_out_path, addon) for addon in ['images', 'weights']]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    TRAIN_PARAMS = {
        'batch_size': 64,
        'epochs': 20,
        'd_lr': 1e-4,
        'g_lr': 1e-4
    }

    gan = MNISTGAN(GAN_PARAMS)
    gan.train(data_loader, model_out_path, TRAIN_PARAMS)
    gan.save_model(model_out_path)


def infer_GAN_MNIST(model_out_path):
    from Models.GAN import MNISTGAN

    p = os.path.join(model_out_path, 'inference')
    if not os.path.exists(p):
            os.makedirs(p)

    params = json.load(open(os.path.join(model_out_path, 'params.json'), 'r'))
    gan = MNISTGAN(params)
    gan.load_model(model_out_path)

    noise = tf.random.normal([100, params['latent_dim']])
    gan.generate(noise, os.path.join(os.path.join(model_out_path, 'inference'), 'inference1.png'))

#####


def train_CGAN_MNIST(model_out_path):
    from Models.CGAN import MNISTCGAN

    GAN_PARAMS = {
        'type': 'CGAN',
        'n_classes': 10,
        'latent_dim': 150,

        'attention': True,
        'attention_additive': True,
        'attention_gamma_init': 1.0,

        'gan_loss': 'CE',   # Hinge, LS, CE

    }

    TRAIN_PARAMS = {
        'batch_size': 128,
        'epochs': 10,
        'd_lr': 1e-4,
        'g_lr': 1e-4
    }
    from Data.MNIST.mnist_data_loader import MNIST_LOADER
    data_loader = MNIST_LOADER()
    print("Data Done")

    paths = [os.path.join(model_out_path, addon) for addon in ['images', 'weights']]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    gan = MNISTCGAN(GAN_PARAMS)
    gan.train(data_loader, model_out_path, TRAIN_PARAMS)
    gan.save_model(model_out_path)


def infer_CGAN_MNIST(model_out_path):

    from Models.CGAN import MNISTCGAN

    p = os.path.join(model_out_path, 'inference')
    if not os.path.exists(p):
        os.makedirs(p)

    params = json.load(open(os.path.join(model_out_path, 'params.json'), 'r'))
    gan = MNISTCGAN(params)
    gan.load_model(model_out_path)

    noise = tf.random.normal([10*params['n_classes'], params['latent_dim']])
    labels = np.repeat(np.arange(params['n_classes']), 10, axis=0).reshape(-1, 1)
    gan.generate(noise, labels, os.path.join(os.path.join(model_out_path, 'inference'), 'inference1.png'))


#####

def train_ScrabbleGAN(model_out_path):

    loader_params = {
        'is_dynamic': True,
        'image_height': 32,
        'char_spatial_range': 16,
        'min_char_length': 10,
        'max_char_length': 18,

        'small_data': False,
        # 'small_data': True,
        # 'small_data_ratio': 0.0005,


        'train_test_split': False,
        'batch_size': 32,
    }

    from Data.IAM.iam_data_loader import IAM_LOADER
    iam_loader = IAM_LOADER(**loader_params)

    crnn_params = {
        'is_dynamic': True,
        'image_height':32,
        'all_chars': iam_loader.all_chars
    }


    scrabble_gan_params = {
        'type': 'ScrabbleGAN',
        'emb_dim': 8192,
        'noise_dim': 32,
        'crnn_params': crnn_params,

        'attention': True,
        'attention_additive': True,
        'attention_gamma_init': 1.0,

        'bn': 'cbn',
        # 'bn': 'regular',

        'gan_loss': 'Hinge' # 'CE', 'LS'
    }

    TRAIN_PARAMS = {
        'batch_size': 32,
        'epochs': 100,
        'd_lr': 2e-4,
        'g_lr': 2e-4,
        'r_lr': 2e-4,

        'gamma': 1.0,
        'gradient_balancing': True,
        'alpha': 1.0,

        'critic_iterations_per_generator': 1,
    }

    from Models.ScrabbleGAN import ScrabbleGAN

    paths = [os.path.join(model_out_path, addon) for addon in ['weights', 'images']]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    scrabble_gan = ScrabbleGAN(scrabble_gan_params)
    scrabble_gan.train(iam_loader, TRAIN_PARAMS, model_out_path)


def infer_ScrabbleGAN(model_out_path):

    from Models.ScrabbleGAN import ScrabbleGAN

    p = os.path.join(model_out_path, 'inference')
    if not os.path.exists(p):
        os.makedirs(p)

    params = json.load(open(os.path.join(model_out_path, 'params.json'), 'r'))
    gan = ScrabbleGAN(params)
    gan.load_model(model_out_path)

    fixed_labels_arrays = [['are', 'You', 'mad', 'sad', 'but', 'bit', 'Tim', 'Tam'],
                           ['hero', 'love', 'need', 'guts', 'time', 'luck', 'must', 'wood'],
                           ['Hello', 'Heavy', 'Sight', 'jolly', 'pasta', 'toast', 'Smile', 'Cross'],
                           ['belief', 'Castle', 'Bishop', 'Knight', 'Fisher', 'Morphy', 'Parrot', 'carrot'],
                           ['Airline', 'alcohol', 'holiday', 'libarty', 'Nuclear', 'Protein', 'Tension', 'display']]
    fixed_noise_arrays = [tf.random.normal([len(fixed_labels_arrays[0]), params['noise_dim']*4]) for _ in range(len(fixed_labels_arrays))]
    gan.generate(fixed_labels_arrays, fixed_noise_arrays, os.path.join(model_out_path, 'inference'))
    gan.generate_single('shalom', os.path.join(model_out_path, 'inference'))

######



def train_infoGAN_MNIST(model_out_path):
    from Data.MNIST.mnist_data_loader import MNIST_LOADER
    from Models.infoGAN import infoGAN

    GAN_PARAMS = {
        'latent_dim': 62,
        'n_classes': 10,
        'n_continuous_codes': 1,
    }

    TRAIN_PARAMS = {
        'batch_size': 64,
        'epochs': 50,
        'd_lr': 2e-4,
        'g_lr': 2e-4,
        'q_lr': 2e-4


    }

    data_loader = MNIST_LOADER()
    print("Data Done")

    paths = [os.path.join(model_out_path, addon) for addon in ['images', 'gen', 'disc']]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    gan = infoGAN(GAN_PARAMS)
    gan.train(data_loader, model_out_path, TRAIN_PARAMS)
    gan.save_model(model_out_path)


####

def train_LatentExplorer(gan_path):
    from Models.ExploreLatent import GanLatentDiscover


    epochs = 100
    steps_per_epoch = 1000
    batch_size = 32
    k = 20

    explainer_path = os.path.join(gan_path, 'latent_explorer')
    paths = [os.path.join(explainer_path, addon) for addon in ['weights', 'images']]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    full_model_params = json.load(open(os.path.join(gan_path, 'params.json'), 'r'))

    if full_model_params['type'] == 'GAN':
        from Models.GAN import MNISTGAN
        gan = MNISTGAN(full_model_params)
        gan.load_model(gan_path)
        G_model = gan.G

    elif full_model_params['type'] == 'CGAN':
        from Models.CGAN import MNISTCGAN
        gan = MNISTCGAN(full_model_params)
        gan.load_model(gan_path)
        G_model = gan.G

    elif full_model_params['type'] == 'ScrabbleGAN':
        from Models.ScrabbleGAN import ScrabbleGAN
        gan = ScrabbleGAN(full_model_params)
        gan.load_model(gan_path)
        G_model = gan.G

    else:
        return

    model = GanLatentDiscover(G_model, full_model_params['type'], k)
    model.train(explainer_path, epochs, steps_per_epoch, batch_size)


def AttentionExplorer(gan_path, data_type):

    from Models.ExploreAttention import get_attention_maps

    explainer_path = os.path.join(gan_path, 'attention_explorer')
    if not os.path.exists(explainer_path):
        os.makedirs(explainer_path)

    full_model_params = json.load(open(os.path.join(gan_path, 'params.json'), 'r'))
    labels = get_labels(is_random=False, n_samples=10, type=full_model_params['type'])

    if full_model_params['type'] == 'GAN':
        from Models.GAN import MNISTGAN
        noise = tf.random.normal([10, full_model_params['latent_dim']])
        m = MNISTGAN
        inputs = noise
    elif full_model_params['type'] == 'CGAN':
        from Models.CGAN import MNISTCGAN
        noise = tf.random.normal([10, full_model_params['latent_dim']])
        m = MNISTCGAN
        inputs = [noise, labels] if labels is not None else noise

    elif full_model_params['type'] == 'ScrabbleGAN':
        noise = tf.random.normal([10, full_model_params['noise_dim']*4])
        from Models.ScrabbleGAN import ScrabbleGAN
        m = ScrabbleGAN
        inputs = [labels, noise] if labels is not None else noise

    else:
        return

    gan = m(full_model_params)
    gan.load_model(gan_path)
    full_model, att_model, att_layer = gan.G, gan.G_att_model, gan.G_att_layer

    att_maps, points = get_attention_maps(att_model, att_layer, inputs, data_type)
    org_images = full_model(inputs)
    print(org_images.shape)

    if full_model_params['type'] == 'ScrabbleGAN':
        org_images = org_images.numpy()[:, :, :, 0]
        org_images = np.transpose(org_images, axes=(0, 2, 1))
        save_act_maps(org_images, points, att_maps, os.path.join(explainer_path, 'attention_maps.png'), rows=10, cols=16)

    else:
        save_act_maps(org_images, points, att_maps, os.path.join(explainer_path, 'attention_maps.png'))


######
"""
def train_InfoScrabble(model_out_path):


    loader_params = {
        'is_dynamic': True,
        'image_height': 32,
        'char_spatial_range': 16,
        'min_char_length': 10,
        'max_char_length': 18,

        # 'small_data': False,
        'small_data': True,
        'small_data_ratio': 0.00005,

        'train_test_split': False,
        'batch_size': 32,
    }
    from Data.IAM.iam_data_loader import RECOGNITION_IAM_LOADER
    iam_loader = RECOGNITION_IAM_LOADER(**loader_params)

    crnn_params = {
        'is_dynamic': loader_params['is_dynamic'],
        'is_recurrent': False,
        'decoder_type': 'tf_beam'
    }

    if crnn_params['is_dynamic']:
        crnn_params['image_height'] = loader_params['image_height']
    else:
        crnn_params['image_size'] = loader_params['image_size'] + [1]
        crnn_params['char_spatial_range'] = 16,
    crnn_params['all_chars'] = iam_loader.all_chars
    if crnn_params['decoder_type'] == 'word_beam':
        crnn_params['corpus'] = iam_loader.corpus

    scrabble_gan_params = {
        'type': 'ScrabbleGAN',
        'emb_dim': 8192,
        'noise_dim': 32,
        'crnn_params': crnn_params,
        'upsampling_method': 'upsampling',
        'attention':True,
        'attention_additive': True,

        'n_classes': 8,
        'n_continuous_codes': 2,

    }

    TRAIN_PARAMS = {
        'batch_size': 32,
        'epochs': 300,
        'd_lr': 2e-4,
        'g_lr': 2e-4,
        'r_lr': 2e-4,
        'q_lr': 2e-4
    }


    paths = [os.path.join(model_out_path, addon) for addon in ['weights', 'images']]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    scrabble_gan = InfoScrabbleGAN(scrabble_gan_params)
    scrabble_gan.train(iam_loader, TRAIN_PARAMS, model_out_path)


"""


def main(parser):
    # parser
    model_path = os.path.join('Results', parser.data, parser.function, parser.model_folder)

    if parser.explore_latent:
        train_LatentExplorer(os.path.join(model_path))
        return

    if parser.explore_attention:
        AttentionExplorer(os.path.join(model_path), parser.data)
        return

    if parser.data == 'MNIST':
        if parser.function == 'GAN':
            if parser.action == 'train':
                train_GAN_MNIST(model_path)
            if parser.action == 'infer':
                infer_GAN_MNIST(model_path)

        elif parser.function == 'CGAN':
            if parser.action == 'train':
                train_CGAN_MNIST(model_path)
            if parser.action == 'infer':
                infer_CGAN_MNIST(model_path)

        elif parser.function == 'InfoGAN':
            if parser.action == 'train':
                train_infoGAN_MNIST(model_path)
            # if parser.action == 'infer':
            #     infer_CGAN_MNIST(model_path)

        elif parser.function == 'ExploreLatent':
            if parser.action == 'train':
                train_CGAN_MNIST(model_path)
            if parser.action == 'infer':
                infer_CGAN_MNIST(model_path)

    elif parser.data == 'IAM':
        if parser.function == 'GAN':
            if parser.action == 'train':
                train_ScrabbleGAN(model_path)
            else:
                infer_ScrabbleGAN(model_path)
        # if parser.function == 'InfoGAN':
        #     if parser.action == 'train':
        #         train_InfoScrabble(model_path)
        #     else:
        #         infer_ScrabbleGAN(model_path)


if __name__ == '__main__':
    import tensorflow as tf
    tf.keras.backend.set_floatx('float32')

    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, choices=['GAN', 'CGAN', 'InfoGAN', ])
    parser.add_argument('--data', type=str, choices=['MNIST', 'IAM', 'IAM_online'])
    parser.add_argument('--action', type=str, choices=['train', 'infer'])

    parser.add_argument('--explore_latent', dest='explore_latent', action='store_true')
    parser.set_defaults(explore_latent=False)

    parser.add_argument('--explore_attention', dest='explore_attention', action='store_true')
    parser.set_defaults(explore_latent=False)


    parser.add_argument('--model_folder', type=str)
    args = parser.parse_args()

    main(args)
