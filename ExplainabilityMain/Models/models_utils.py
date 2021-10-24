import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, Conv2D, Conv2DTranspose, UpSampling2D, Softmax, ReLU, Add, AveragePooling2D, Flatten
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.layers import Activation, Layer
from tensorflow_addons.layers import SpectralNormalization
import numpy as np

# init = tf.keras.initializers.Orthogonal()
init = tf.keras.initializers.RandomNormal(stddev=0.02)


class ConditionalBatchNorm(Layer):
    """CBN layer as described in https://arxiv.org/pdf/1707.00683.pdf"""
    def __init__(self, channels):
        super(ConditionalBatchNorm, self).__init__()

        self.channels = channels
        self.gamma_layer = Dense(units=channels, use_bias=False)
        self.beta_layer = Dense(units=channels, use_bias=False)
        self.batch_clean = BatchNormalization(scale=False, center=False)

    def call(self, inputs, **kwargs):

        images = inputs[0]
        noise_conditioning = Flatten()(inputs[1])
        out = self.batch_clean(images)

        gamma = self.gamma_layer(noise_conditioning)
        gamma = tf.expand_dims(gamma, axis=1)
        gamma = tf.expand_dims(gamma, axis=1)

        out *= gamma

        beta = self.beta_layer(noise_conditioning)
        beta = tf.expand_dims(beta, axis=1)
        beta = tf.expand_dims(beta, axis=1)

        out += beta

        return out

    def get_config(self):
        return {"channels": self.channels}


class SpatialEmbedding(Layer):

    def __init__(self, vocab_size, filter_dim):
        super(SpatialEmbedding, self).__init__(name='SpatialEmbedding')
        self.vocab_size = vocab_size
        self.filter_dim = filter_dim

    def build(self, input_shape):
        self.kernel = self.add_weight("filter_bank",
                                      shape=[self.vocab_size] + list(self.filter_dim),
                                      trainable=True)

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.kernel, inputs)

    def get_config(self):
        config = super(SpatialEmbedding, self).get_config()
        config.update({'vocab_size': self.vocab_size, 'filter_dim': self.filter_dim})
        return config


class TripleConv(Layer):
    def __init__(self, channels, kernel_size, stride, weights_normalization=None, normalization=None, activation=None):
        super(TripleConv, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weights_normalization = weights_normalization
        self.normalization = normalization
        self.activation = activation

        self.layers = []
        if weights_normalization == 'sn':
            conv_layer = SpectralNormalization(Conv2D(channels, kernel_size, stride, padding='same', kernel_initializer=init))
        else:
            conv_layer = Conv2D(channels, kernel_size, stride, padding='same', kernel_initializer=init)

        self.layers.append(conv_layer)

        if normalization is not None:
            if normalization == 'bn':
                self.layers.append(BatchNormalization())

        if activation is not None:
            activation_layer = get_activation(activation)
            self.layers.append(activation_layer)

    def call(self, inputs, *args, **kwargs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        return {
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'weights_normalization': self.weights_normalization,
            'normalization': self.normalization,
            'activation': self.activation,
        }


class TripleTConv(Layer):
    def __init__(self, channels, kernel_size, stride, weights_normalization=None, normalization=None, activation=None):
        super(TripleTConv, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weights_normalization = weights_normalization
        self.normalization = normalization
        self.activation = activation

        self.layers = []
        if weights_normalization == 'sn':
            conv_layer = SpectralNormalization(Conv2DTranspose(channels, kernel_size, stride, padding='same', kernel_initializer=init))
        else:
            conv_layer = Conv2DTranspose(channels, kernel_size, stride, padding='same', kernel_initializer=init)

        self.layers.append(conv_layer)

        if normalization is not None:
            if normalization == 'bn':
                self.layers.append(BatchNormalization())

        if activation is not None:
            activation_layer = get_activation(activation)
            self.layers.append(activation_layer)

    def call(self, inputs, *args, **kwargs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        return {
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'weights_normalization': self.weights_normalization,
            'normalization': self.normalization,
            'activation': self.activation,
        }


class TripleDense(Layer):
    def __init__(self, size, weights_normalization, normalization, activation):
        super(TripleDense, self).__init__()

        self.size = size
        self.weights_normalization = weights_normalization
        self.normalization = normalization
        self.activation = activation

        self.layers = []
        if weights_normalization == 'sn':
            dense_layer = SpectralNormalization(Dense(size, kernel_initializer=init))
        else:
            dense_layer = Dense(size, kernel_initializer=init)

        self.layers.append(dense_layer)

        if normalization == 'bn':
            self.layers.append(BatchNormalization())

        if activation != None:
            activation_layer = get_activation(activation)
            self.layers.append(activation_layer)

    def call(self, inputs, *args, **kwargs):

        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        return {
            'size': self.size,
            'weights_normalization': self.weights_normalization,
            'normalization': self.normalization,
            'activation': self.activation,
        }


class ResBlock(Layer):
    def __init__(self, out_ch, kernel_sizes, weights_normalization=None):
        super(ResBlock, self).__init__()
        self.out_ch = out_ch
        self.weights_normalization = weights_normalization

        self.x_layers = []

        self.x_layers.append(BatchNormalization())

        self.x_layers.append(get_activation('relu'))
        self.x_layers.append(TripleConv(channels=out_ch, kernel_size=kernel_sizes[0], stride=(1, 1), weights_normalization=weights_normalization))

        self.x_layers.append(get_activation('relu'))
        self.x_layers.append(TripleConv(channels=out_ch, kernel_size=kernel_sizes[1], stride=(1, 1), weights_normalization=weights_normalization))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x_skip = x

        for layer in self.x_layers:
            x = layer(x)

        x = Add()([x, x_skip])
        return x

    def get_config(self):
        return {
            'out_ch': self.out_ch,
            'weights_normalization': self.weights_normalization,
        }


class ResUpBlock(Layer):
    def __init__(self, resolution, in_ch, out_ch, kernel_sizes, batch_normalization_type, upsampling_type,  weights_normalization=None):
        super(ResUpBlock, self).__init__()

        self.resolution = resolution
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.batch_normalization_type = batch_normalization_type
        self.upsampling_type = upsampling_type
        self.weights_normalization = weights_normalization


        self.x_skip_layers = []
        self.x_layers = []
        self.batch_normalization_type = batch_normalization_type

        ## skip

        if upsampling_type == 'upsampling':
            self.x_skip_layers.append(UpSampling2D(size=resolution))
            self.x_skip_layers.append(TripleConv(channels=out_ch, kernel_size=1, stride=(1, 1), weights_normalization=weights_normalization))
        else:
            self.x_skip_layers.append(TripleTConv(channels=out_ch, kernel_size=1, stride=resolution, weights_normalization=weights_normalization))

        # main

        if batch_normalization_type == 'regular':
            self.x_layers.append(BatchNormalization())
        else:
            self.x_layers.append(ConditionalBatchNorm(channels=in_ch))

        self.x_layers.append(get_activation('relu'))

        if upsampling_type == 'upsampling':
            self.x_layers.append(UpSampling2D(size=resolution))
            self.x_layers.append(TripleConv(channels=out_ch, kernel_size=kernel_sizes[0], stride=(1, 1), weights_normalization=weights_normalization))
        elif upsampling_type == 't_conv':
            self.x_layers.append(TripleTConv(channels=out_ch, kernel_size=kernel_sizes[1], stride=resolution, weights_normalization=weights_normalization))
        else:
            raise NotImplemented

        if batch_normalization_type == 'regular':
            self.x_layers.append(BatchNormalization())
        else:
            self.x_layers.append(ConditionalBatchNorm(channels=out_ch))

        self.x_layers.append(get_activation('relu'))

        self.x_layers.append(TripleConv(channels=out_ch, kernel_size=1, stride=(1, 1), weights_normalization=weights_normalization))

    def call(self, inputs, *args, **kwargs):
        if self.batch_normalization_type == 'regular':
            x = inputs
            x_skip = inputs

            for layer in self.x_layers:
                x = layer(x)
            for layer in self.x_skip_layers:
                x_skip = layer(x_skip)

            f = x_skip + x
            return f

        else:
            x = inputs[0]
            x_skip = inputs[0]
            noise = inputs[1]

            for layer in self.x_layers:
                if layer.name.startswith('conditional_batch_norm'):
                    x = layer([x, noise])
                else:
                    x = layer(x)

            for layer in self.x_skip_layers:
                x_skip = layer(x_skip)

            f = x_skip + x
            return f

    def get_config(self):
        return {
            'resolution': self.resolution,
            'in_ch': self.in_ch,
            'out_ch': self.out_ch,
            'batch_normalization_type': self.batch_normalization_type,
            'upsampling_type': self.upsampling_type,
            'weights_normalization': self.weights_normalization,
        }


class ResDownBlock(Layer):
    def __init__(self, resolution, kernel_sizes, out_ch, weights_normalization=None):
        super(ResDownBlock, self).__init__()

        self.x_skip_layers = []
        self.x_layers = []

        self.x_skip_layers.append(TripleConv(channels=out_ch, kernel_size=1, stride=(1, 1), weights_normalization=weights_normalization))
        self.x_skip_layers.append(AveragePooling2D(pool_size=resolution))

        self.x_layers.append(get_activation('l_relu'))
        self.x_layers.append(TripleConv(channels=out_ch, kernel_size=kernel_sizes[0], stride=(1, 1), weights_normalization=weights_normalization))


        self.x_layers.append(get_activation('l_relu'))
        self.x_layers.append(TripleConv(channels=out_ch, kernel_size=kernel_sizes[1], stride=(1, 1), weights_normalization=weights_normalization))

        self.x_layers.append(AveragePooling2D(pool_size=resolution))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x_skip = inputs

        for layer in self.x_layers:
            x = layer(x)

        for layer in self.x_skip_layers:
            x_skip = layer(x_skip)

        f = Add()([x_skip, x])
        return f

    def get_config(self):
        return {
            'resolution': self.resolution,
            'out_ch': self.out_ch,
            'weights_normalization': self.weights_normalization,
        }


class SelfAttentionLayer(Layer):
    def __init__(self, ch, add_input, attention_gamma_init, name=None):
        if name is not None:
            super(SelfAttentionLayer, self).__init__(name=name)
        else:
            super(SelfAttentionLayer, self).__init__()

        self.add_input = add_input
        self.ch = ch
        self.f = SpectralNormalization(Conv2D(ch//8, 1, padding='same', kernel_initializer=tf.initializers.orthogonal()))
        self.g = SpectralNormalization(Conv2D(ch//8, 1, padding='same', kernel_initializer=tf.initializers.orthogonal()))
        self.h = SpectralNormalization(Conv2D(ch, 1, padding='same', kernel_initializer=tf.initializers.orthogonal()))

        if self.add_input:
            self.gamma = self.add_weight(shape=(), initializer=tf.constant_initializer(attention_gamma_init), trainable=True, name='gamma')

    def call(self, inputs, *args, **kwargs):
        shape = tf.shape(inputs)
        q, k, v = self.f(inputs), self.g(inputs), self.h(inputs)

        q = tf.reshape(q, (shape[0], shape[1]*shape[2], self.ch//8))
        k = tf.reshape(k, (shape[0], shape[1]*shape[2], self.ch//8))
        v = tf.reshape(v, (shape[0], shape[1]*shape[2], self.ch))

        s = tf.matmul(q, k, transpose_b=True)
        s = Softmax()(s)
        o = tf.matmul(s, v)

        o = tf.reshape(o, shape)

        if self.add_input:
            o = tf.add(tf.multiply(self.gamma,  o), inputs)

        return o

    def get_config(self):
        return {
            'ch': self.ch,
            'add_input': self.add_input,
        }


def get_activation(activation):
    if activation == 'relu':
        return ReLU()
    elif activation == 'l_relu':
        return LeakyReLU()
    elif activation == 'sigmoid':
        return Activation(sigmoid)
    elif activation == 'softmax':
        return Softmax()
    elif activation == 'tanh':
        return Activation(tanh)


def sparsify(texts, char_list):
    """Put ground truth texts into sparse tensor for ctc_loss."""
    indices = []
    values = []
    shape = [len(texts), 0]  # last entry must be max(labelList[i])

    # go over all texts
    for batchElement, text in enumerate(texts):
        # convert to string of label (i.e. class-ids)
        label_str = [char_list.index(c) for c in text]
        # sparse tensor must have size of max. label-string
        if len(label_str) > shape[1]:
            shape[1] = len(label_str)
        # put each label into sparse tensor
        for i, label in enumerate(label_str):
            indices.append([batchElement, i])
            values.append(label)

    # sparse_tensor =
    # sparse_tensor = tf.cast(sparse_tensor, dtype=tf.int32)
    return tf.SparseTensor(indices, values, shape)


def get_labels_true_length(raw_labels):
    lengths = [len(label) for label in raw_labels]
    return lengths


def ctc_loss(model_out, sparse_true_labels, true_labels_lengths):
    sparse_true_labels = tf.cast(sparse_true_labels, tf.int32)
    y_pred_shape = tf.shape(model_out)
    logit_length = tf.fill([y_pred_shape[0]], y_pred_shape[1])

    loss = tf.nn.ctc_loss(
        labels=sparse_true_labels,
        logits=model_out,
        label_length=true_labels_lengths,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=-1,
    )
    return tf.math.reduce_mean(tf.boolean_mask(loss, tf.math.is_finite(loss)))