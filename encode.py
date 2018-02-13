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
import shutil
import tensorflow as tf

from model_mp import *
from ops import *
from utils import *

pe = os.path.exists
pj = os.path.join

batch_size = 128
batches_per_epoch = 25
beta1 = 0.5
ckpt_dir = "/home/matt.phillips/Repos/Kitware/DCGAN-tensorflow/checkpoint" 
df_dim = 64
gan_batch_size = 64
im_shape = (None, 64, 64, 3)
learning_rate = 0.001
N = 10000
num_epochs = 10
sample_dir = "/home/matt.phillips/Repos/Kitware/DCGAN-tensorflow/samples" 
z_dim = 100


def build_and_load_dcgan():
    config = tf.ConfigProto() #allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) #, graph=tf.Graph())

    dcgan = DCGAN( \
                sess, 
                input_width=im_shape[2],
                input_height=im_shape[1],
                output_width=im_shape[2],
                output_height=im_shape[1],
                batch_size=gan_batch_size,
                sample_num=gan_batch_size,
                z_dim=z_dim,
                dataset_name="planet",
                input_fname_pattern="*.jpg", #"*.png",
                crop=False,
                checkpoint_dir=ckpt_dir,
                sample_dir=sample_dir)

    show_all_variables()
    dcgan.load(ckpt_dir)
    return dcgan


def build_encoder():
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
        bn1 = batch_norm(name="e_bn1")
        bn2 = batch_norm(name="e_bn2")
        bn3 = batch_norm(name="e_bn3")

        image = tf.placeholder(tf.float32, shape=im_shape, name="image")
        h0 = lrelu( conv2d( image, df_dim, name='e_h0_conv') )
        h1 = lrelu( bn1( conv2d(h0, df_dim*2, name='e_h1_conv') ) )
        h2 = lrelu( bn2( conv2d(h1, df_dim*4, name='e_h2_conv') ) )
        h3 = lrelu( bn3( conv2d(h2, df_dim*8, name='e_h3_conv') ) )
        h3_shape = h3.get_shape().as_list()
        h4 = linear(tf.reshape(h3, [-1, np.prod(h3_shape[1:])]), z_dim, 
                'e_h4_lin')
#        h4 = linear(tf.reshape(h3, [gan_batch_size, -1]), z_dim, 'e_h4_lin')

        return tf.nn.tanh(h4, name="enc_out")


def generate_data(dcgan):
    sess = dcgan.sess
    zs = []
    imgs_hat = []
    print("Generating z-vector, image pairs...")
    for i in range(N // gan_batch_size):
        batch_z = np.random.uniform(-1, 1, size=(gan_batch_size, z_dim))
        gen_out = dcgan.sess.graph.get_tensor_by_name("generator/gen_out:0")
        images = gen_out.eval(session=sess, feed_dict={ dcgan.z : batch_z })
        zs.append(batch_z)
        imgs_hat.append(images)
    print("Done, %d batches generated" % (len(zs)))
    return zs,imgs_hat
    

def make_training_op(sess):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
        enc_out = sess.graph.get_tensor_by_name("encoder/enc_out:0")
        z = tf.placeholder(tf.float32, enc_out.get_shape(), name="z")
        enc_loss = tf.reduce_mean(tf.square(z - enc_out), name="enc_loss")
        t_vars = tf.trainable_variables()
        e_vars = [var for var in t_vars if "e_" in var.name]
        training_op = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                .minimize(enc_loss, var_list=e_vars)
        return training_op,z


def train_encoder(encoder, imgs_hat, zs):
    if len(imgs_hat) != len(zs):
        raise RuntimeError("Different lengths of imgs_hat, zs")
    sess = tf.Session()
    training_op,z = make_training_op(sess)
    sess.run( tf.global_variables_initializer() )
    show_all_variables()
#    for op in sess.graph.get_operations():
#        print(op)

    saver = tf.train.Saver()

    if pe(pj(ckpt_dir, "saved")):
        shutil.rmtree(pj(ckpt_dir, "saved"))
    builder = tf.saved_model.builder.SavedModelBuilder( pj(ckpt_dir, "saved") )
    builder.add_meta_graph_and_variables(sess,
            [tf.saved_model.tag_constants.TRAINING])

#    z = sess.graph.get_tensor_by_name("encoder/z:0")
    image = sess.graph.get_tensor_by_name("encoder/image:0")
    enc_loss = sess.graph.get_tensor_by_name("encoder_1/enc_loss:0")
   
    for ep in range(num_epochs):
        for i in range( len(imgs_hat) ):
            batch_imgs = imgs_hat[i]
            batch_z = zs[i]
            sess.run([training_op], 
                    feed_dict={ image : batch_imgs, z : batch_z })

        loss = enc_loss.eval(session=sess, 
               feed_dict={ image : batch_imgs, z : batch_z })
        print("Loss: %f" % (loss))
        saver.save(sess, pj(ckpt_dir, "encoder", "model_%03d.ckpt" % (ep)))
        builder.save()
    

def main(args):
#    with tf.device("/gpu:0"):
    dcgan = build_and_load_dcgan()
    zs,imgs_hat = generate_data(dcgan)
    encoder = build_encoder()
    train_encoder(encoder, imgs_hat, zs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
