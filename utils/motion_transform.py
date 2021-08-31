import h5py
import numpy as np

import os
import pickle


def motion_transform(motion_data, config):

    with h5py.File(config['file_pos_minmax'], 'r') as f:
        init = np.array(f['init'])
        minmax = np.array(f['minmax']).T
        pos_min = np.floor(minmax[0, :][None, :]) - np.ones_like(minmax[0, :][None, :])
        pos_max = np.floor(minmax[1, :][None, :]) + np.ones_like(minmax[1, :][None, :])

    if config['normalization'] == 'interval':
        div = pos_max - pos_min
        div[div == 0] = 1
        config['slope_pos'] = (config['rng_pos'][1] - config['rng_pos'][0]) / div
        config['intersec_pos'] = config['rng_pos'][1] - config['slope_pos'] * pos_max
        init_trans = np.mean(motion_data[0:26, :], axis=0)
        motion_data[:, :] -= init
        motion_data = motion_data * config['slope_pos'] + config['intersec_pos']

        return motion_data, init_trans

    elif config['normalization'] == 'none':
        '''
        motion transform do nothing...
        Return the exact same data and the means of the first second.
        '''
        init_trans = np.mean(motion_data[0:25, :], axis=0)
        return motion_data, init_trans

    elif config['normalization'] == 'linear':
        init_trans = np.mean(motion_data[0:25, :], axis=0)
        return motion_data/200, init_trans


def reverse_motion_transform(motion_data, config):
    with h5py.File(config['file_pos_minmax'], 'r') as f:
        init = np.array(f['init'])
        minmax = np.array(f['minmax']).T
        pos_min = np.floor(minmax[0, :][None, :]) - np.ones_like(minmax[0, :][None, :])
        pos_max = np.floor(minmax[1, :][None, :]) + np.ones_like(minmax[1, :][None, :])

    if config['normalization'] == 'interval':
        div = pos_max - pos_min
        div[div == 0] = 1
        config['slope_pos'] = (config['rng_pos'][1] - config['rng_pos'][0]) / div
        config['intersec_pos'] = config['rng_pos'][1] - config['slope_pos'] * pos_max
        motion_data = (motion_data - config['intersec_pos']) / config['slope_pos']
        motion_data += init
        return motion_data
    elif config['normalization'] == 'none':
        return motion_data
    elif config['normalization'] == 'linear':
        return motion_data*200


def motion_silence(motion, size):
    pos = np.expand_dims(motion, axis=0)
    silence_pos = np.repeat(pos, size, axis=0)
    #silence_pos = np.ones((size, motion.shape[1]), dtype=np.float32) * motion[0, :]
    return silence_pos


if __name__ == '__main__':
    prefix = os.getcwd().replace('utils', 'exp/data/train/trainf000.h5')
    with h5py.File(prefix, 'r') as f:
        motion = np.array(f['motion'])
    prefix = os.getcwd().replace('utils', 'exp')
    with open(os.path.join(prefix, 'configuration.pickle'), 'rb') as f:
        config = pickle.load(f)
    config['rng_pos'] = [-0.9, 0.9]
    config['rng_wav'] = [-0.9, 0.9]
    print(motion.shape)
    quit()
    motion_transform(motion, config)
