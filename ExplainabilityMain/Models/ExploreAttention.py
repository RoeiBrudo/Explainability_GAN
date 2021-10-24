import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Softmax
import itertools


def get_attention_maps(pre_model, att_layer, batch_input_params_for_model, type):

    last_features = pre_model(batch_input_params_for_model)
    q, k = att_layer.f(last_features), att_layer.g(last_features)

    shape_temp = tf.shape(last_features)
    q = tf.reshape(q, (shape_temp[0], shape_temp[1]*shape_temp[2], att_layer.ch//8))
    k = tf.reshape(k, (shape_temp[0], shape_temp[1]*shape_temp[2], att_layer.ch//8))

    s = tf.matmul(q, k, transpose_b=True)
    s = Softmax()(s)

    s = s.numpy()

    if type == 'MNIST':
        h_points = [7, 14, 21]
        w_points = [7, 14, 21]
        shape = (28, 28)

    else:
        w_points = [w for w in np.linspace(start=32/5, stop=4*32/5, num=3, dtype=np.int)]

        width = 3*16
        h_points = [w for w in np.linspace(start=int(width/5), stop=int(4*width/5), num=5, dtype=np.int)]
        shape = (32, width)

    points_idx = [p[0]*p[1] for p in itertools.product(*[h_points, w_points])]

    maps = []
    for batch_ind in range(10):
        att_maps = s[batch_ind, :, :]
        maps_i = [att_maps[map_ind, :].reshape(shape) for map_ind in points_idx]

        maps.append(maps_i)

    return maps, itertools.product(*[h_points, w_points])








