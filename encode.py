"""
In the simplest incarnation

0. Build the encoder (if necessary)
1. Load DCGAN model
2. Use sampler to create batches
3. Train the encoder
"""

import argparse
import numpy as np
import os
import tensorflow as tf

from model import *
from ops import *
from utils import *

batch_size = 128
df_dim = 64
im_shape = (64,64,3)
z_dim = 100

def build_encoder():
    with tf.variable_scope("encoder") as scope:
        bn1 = batch_norm(name="bn1")
        bn2 = batch_norm(name="bn2")
        bn3 = batch_norm(name="bn3")

        image = tf.placeholder(tf.float32, im_shape, name="image")
        h0 = lrelu( conv2d( image, df_dim, name='e_h0_conv') )
        h1 = lrelu( bn1( conv2d(h0, df_dim*2, name='e_h1_conv') ) )
        h2 = lrelu( bn2( conv2d(h1, df_dim*4, name='e_h2_conv') ) )
        h3 = lrelu( bn3( conv2d(h2, df_dim*8, name='e_h3_conv') ) )
        h4 = linear(tf.reshape(h3, [batch_size, -1]), z_dim, 'e_h4_lin')

        return tf.nn.tanh(h4, name="out")


def main(args):
    dcgan = DCGAN()
    encoder = build_encoder()
    train_encoder(encoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
