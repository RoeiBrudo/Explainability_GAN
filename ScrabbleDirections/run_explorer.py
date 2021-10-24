# import sys
# sys.path.extend(['/home/ubuntu/workspace/scrabble-gan'])

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)

from src.bigacgan.net_architecture import make_generator, make_discriminator, make_recognizer, make_gan
import gin

from src.bigacgan.arch_ops import spectral_norm
from src.bigacgan.data_utils import load_prepare_data, train, make_gif, load_random_word_list
from src.bigacgan.net_architecture import make_generator, make_discriminator, make_recognizer, make_gan
from src.bigacgan.net_loss import hinge, not_saturating

from ExploreLatent import GanLatentDiscover



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



gin.external_configurable(hinge)
gin.external_configurable(not_saturating)
gin.external_configurable(spectral_norm)

from src.dinterface.dinterface import init_reading

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@gin.configurable
def setup_optimizer(g_lr, d_lr, r_lr, beta_1, beta_2, loss_fn, disc_iters):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2)
    recognizer_optimizer = tf.keras.optimizers.Adam(learning_rate=r_lr, beta_1=beta_1, beta_2=beta_2)
    return generator_optimizer, discriminator_optimizer, recognizer_optimizer, loss_fn, disc_iters


@gin.configurable('shared_specs')
def get_shared_specs(epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention):
    return epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention


@gin.configurable('io')
def setup_io(base_path, checkpoint_dir, gen_imgs_dir, model_dir, raw_dir, read_dir, input_dim, buf_size, n_classes,
             seq_len, char_vec, bucket_size):
    gen_path = base_path + gen_imgs_dir
    ckpt_path = base_path + checkpoint_dir
    m_path = base_path + model_dir
    raw_dir = base_path + raw_dir
    read_dir = base_path + read_dir
    return input_dim, buf_size, n_classes, seq_len, bucket_size, ckpt_path, gen_path, m_path, raw_dir, read_dir, char_vec

def load_model():
    path_to_saved_model = 'res/out/big_ac_gan/model/generator_15'
    gin.parse_config_file(os.path.join('src', 'scrabble_gan.gin'))
    epochs, batch_size, latent_dim, embed_y, num_gen, kernel_reg, g_bw_attention, d_bw_attention = get_shared_specs()
    in_dim, buf_size, n_classes, seq_len, bucket_size, ckpt_path, gen_path, m_path, raw_dir, read_dir, char_vec = setup_io()

    imported_model = make_generator(latent_dim, in_dim, embed_y, gen_path, kernel_reg, g_bw_attention, n_classes)
    imported_model.load_weights(path_to_saved_model)
    print("Done!")
    return imported_model


if __name__ == '__main__':
    g_model = load_model()
    explorer = GanLatentDiscover(g_model, 'ScrabbleGAN', 10)
    explorer.train(os.path.join('out', 'explorer'), epochs=2, steps_per_epoch=2, batch_size=3)