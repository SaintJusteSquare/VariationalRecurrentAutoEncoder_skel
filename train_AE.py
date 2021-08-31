#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tf.logging.set_verbosity(tf.logging.DEBUG)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import sys

module_utils = os.path.join(os.getcwd(), 'utils')
sys.path.append(module_utils)
from utils.dataset import Skeleton
from utils.plot_result import test_draw, draw_image, draw
from motion_transform import reverse_motion_transform

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
from networks.model import Vrnn, Sampling

#############################
# Settings
#############################

BATCH_SIZE = 128
time_steps = 50

exp = 'exp2'
model_name = os.path.join(exp, 'variationalRNN+GRU1')
data_path = 'data'
configuration = {'file_pos_minmax': 'data/pos_minmax.h5',
                 'normalization': 'interval',
                 'rng_pos': [-0.9, 0.9]}

if not os.path.isdir(exp):
    os.makedirs(exp)
if not os.path.isdir(model_name):
    os.makedirs(model_name)

################################
# Loading datasets (Train/Test)
################################

train_path = os.path.join(data_path, 'train')
train_generator = Skeleton(train_path, 'train', configuration, BATCH_SIZE, sequence=time_steps, init_step=1,
                           shuffle=True)
x_train = train_generator.get_dataset(type='float32')

test_path = os.path.join(data_path, 'test')
test_generator = Skeleton(test_path, 'test', configuration, BATCH_SIZE, sequence=time_steps, init_step=1, shuffle=True)
x_test = train_generator.get_dataset(type='float32')

print('x_train.shape: ', x_train.shape)
print('x_train mean: ', np.mean(x_train))
print('x_test.shape: ', x_test.shape)
print('x_test mean: ', np.mean(x_test))

#############################
# Helper functions
#############################

"""
def reconstruct_sequence(model, test_sequence, exp, export_to_file=True, name='test_sequence'):
    predictions = model.predict_on_batch(test_sequence).numpy()
    predictions = np.reshape(predictions, (time_steps, 69))
    predictions = reverse_motion_transform(predictions, configuration)
    predictions = np.reshape(predictions, (time_steps, 23, 3))
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # opencv3.0
    name = os.path.join(exp, name)
    videoWriter = cv2.VideoWriter(name + '.avi', fourcc, 25, (600, 400))
    draw(predictions, export_to_file=export_to_file, videoWriter_enable=videoWriter)
    videoWriter.release()
    cv2.destroyAllWindows()
"""


def draw_sequence(test_sequence, exp_folder, export_to_file=True, name='sequence'):
    size = test_sequence.shape[0]
    test_sequence = np.reshape(test_sequence, (size, 23, 3))
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # opencv3.0
    name = os.path.join(exp_folder, name)
    videoWriter = cv2.VideoWriter(name + '.avi', fourcc, 25, (600, 400))
    draw(test_sequence, export_to_file=export_to_file, videoWriter_enable=videoWriter)
    videoWriter.release()
    cv2.destroyAllWindows()


def draw_sequece_from_state(random_state, sequence, nom):
    sample_sequence = model.sample_from_state(random_state, sequence)
    sample_sequence = np.squeeze(sample_sequence)
    sample_sequence = reverse_motion_transform(sample_sequence, configuration)
    draw_sequence(sample_sequence, exp_folder=model_name, name=nom)


def draw_sequece_from_distributions(initial_frame, distributions, nom):
    sample_sequence = model.sample(initial_frame, distributions)
    sample_sequence = np.squeeze(sample_sequence)
    sample_sequence = reverse_motion_transform(sample_sequence, configuration)
    draw_sequence(sample_sequence, exp_folder=model_name, name=nom)


def plot_and_save_loss(train_loss, name, val_loss=None):
    plt.plot(train_loss)
    if val_loss is not None:
        plt.plot(val_loss)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
    else:
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
    plt.savefig(os.path.join(name, 'loss_history.png'))


def plot_comparison(comparison, exp, name):
    fig, ax = plt.subplots()
    ax.plot(comparison)
    ax.set_xlabel('diff')
    ax.set_ylabel('timestep')
    ax.figure.savefig(os.path.join(exp, name))
    plt.close('all')


#############################
# Tensorflow dataset
#############################

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(BATCH_SIZE).shuffle(x_train.shape[0])
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE).shuffle(x_test.shape[0])

#############################
# Models/training
#############################

epochs = 50

x_dim = 69
x2s_dim = 50
z_dim = 50
z2s_dim = 20

k = 1
h_dim = 1000

q_z_dim = 150
p_z_dim = 150
p_x_dim = 150

batch_size = BATCH_SIZE

initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([50], [1e-4, 1e-5])
optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

loss_metric = tf.keras.metrics.Mean()

model = Vrnn(x_dim=x_dim, x2s_dim=x2s_dim, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim, q_z_dim=q_z_dim,
             p_z_dim=p_z_dim, p_x_dim=p_x_dim, mode='gauss', k=k)
sampler = Sampling()

folder_images = os.path.join(exp, 'folder_image')
if not os.path.isdir(folder_images):
    os.makedirs(folder_images)

groundtruth_0 = x_train[0]
first_frame = np.expand_dims(groundtruth_0[0], axis=0)

groundtruth_0 = np.expand_dims(groundtruth_0, axis=0)
test_theta_mu, test_theta_sig, test_z = model(groundtruth_0)
reconstruction = sampler.sample_sequence([test_theta_mu, test_theta_sig])
reconstruction = np.squeeze(reconstruction)
reconstruction = reverse_motion_transform(reconstruction, configuration)
nom = 'reconstruction_at_epoch_{:04d}.png'.format(0)
draw_sequence(reconstruction, exp_folder=model_name, name=nom)

fromDistrib = model.sample(first_frame, test_z)
fromDistrib = np.squeeze(fromDistrib)
fromDistrib = reverse_motion_transform(fromDistrib, configuration)
nom = 'Sample_from_z_at_epoch{:04d}.png'.format(0)
draw_sequence(fromDistrib, exp_folder=model_name, name=nom)

groundtruth_0 = np.squeeze(groundtruth_0)
groundtruth_0 = reverse_motion_transform(groundtruth_0, configuration)
nom = 'groundtruth_sequence'
draw_sequence(groundtruth_0, exp_folder=model_name, name=nom)

first_frame = np.expand_dims(groundtruth_0[0], axis=0)
groundtruth_0 = np.expand_dims(groundtruth_0, axis=0)

epochs_list = list()
loss_list = list()
# val_loss_list = list()
for epoch in range(1, epochs + 1):
    print("Start of epoch %d" % (epoch,))
    start_time = time.time()
    for step, train_x in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = model(train_x)
            loss = model.losses

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer1.apply_gradients(zip(grads, model.trainable_weights))

        loss_metric(loss)
        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
        if step == 300:
            break

    end_time = time.time()

    if epoch % 3 == 0:
        print('Epoch: {}, train: {}, ''time elapse for current epoch {}'.format(epoch, loss_metric.result(),
                                                                                end_time - start_time))

        groundtruth_0 = x_train[0]
        first_frame = np.expand_dims(groundtruth_0[0], axis=0)
        groundtruth_0 = np.expand_dims(groundtruth_0, axis=0)
        test_theta_mu, test_theta_sig, test_z = model(groundtruth_0)
        reconstruction = sampler.sample_sequence([test_theta_mu, test_theta_sig])
        reconstruction = np.squeeze(reconstruction)
        reconstruction = reverse_motion_transform(reconstruction, configuration)
        nom = 'reconstruction_at_epoch_{:04d}.png'.format(epoch)
        draw_sequence(reconstruction, exp_folder=model_name, name=nom)

        fromDistrib, state_list, zprime_list = model.sample(first_frame, test_z, return_state=True)
        fromDistrib = np.squeeze(fromDistrib)
        fromDistrib = reverse_motion_transform(fromDistrib, configuration)
        nom = 'Sample_from_z_at_epoch{:04d}.png'.format(epoch)
        draw_sequence(fromDistrib, exp_folder=model_name, name=nom)
        epochs_list.append(epoch)
        loss_list.append(loss_metric.result())

        zprime_list = np.array(zprime_list)
        zprime_comparison = model.comparison(zprime_list, with_what='zprime')
        plot_comparison(zprime_comparison, exp=model_name, name='zprime_comparison_at_epoch{:04d}.png'.format(epoch))

        state_list = np.array(state_list)
        state_comparison = model.comparison(state_list, with_what='state')
        plot_comparison(state_comparison, exp=model_name, name='state_comparison_at_epoch{:04d}.png'.format(epoch))

plot_and_save_loss(loss_list, model_name)

"""
sequence_0 = x_train[0][None, :]
reconstruct_sequence(model, sequence_0, model_name, name='train_sequence')

train_file_path = './data/train/trainf030.h5'
test_file_path = './data/test/testf000.h5'
name, data, config = train_generator.get_file_data(train_file_path)
name = name.split('/')[-1]
data = reverse_motion_transform(data, configuration)
print('name: ', name)
print('data shape: ', data.shape)
print('data mean: ', np.mean(data))

draw_sequence(data, model_name, name=name)

frames = data.shape[0]concatenate([z_1_temp, s_temp])

_data = np.reshape(data, (1, frames, 69))
predictions = model.predict_on_batch(_data).numpy()
predictions = np.reshape(predictions, (frames, 69))
predictions = reverse_motion_transform(predictions, configuration)
predictions = np.reshape(predictions, (frames, 23, 3))

pred_name = name + '_reconstructed'

        print('\n')
        print("Last transformations : ")
draw_sequence(predictions, model_name, name=pred_name)

_data = data[:200]
_data = np.reshape(_data, (1, 200, 69))
predictions = model.predict_on_batch(_data).numpy()
predictions = np.reshape(predictions, (200, 69))
predictions = reverse_motion_transform(predictions, configuration)
predictions = np.reshape(predictions, (200, 23, 3))

pred_name = name + '_partialy_reconstructed'
draw_sequence(predictions, model_name, name=pred_name)
"""
