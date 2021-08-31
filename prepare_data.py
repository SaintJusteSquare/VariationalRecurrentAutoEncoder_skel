import os
import glob
import h5py
import numpy as np

import sys

module_utils = os.path.join(os.getcwd(), 'utils')
sys.path.append(module_utils)

from utils.motion_loader import output_loader
from utils.motion_transform import motion_transform

dataset_master = './dataset_master'
exp = './data_untransformed'
configuration = {'step': 0, 'fps': 25, 'normalization': 'none', 'rng_pos': [-0.9, 0.9]}


def calculate_minmax(fileslist):
    if len(fileslist) < 1:
        raise ValueError()
    init = 0
    for item in fileslist:
        with h5py.File(item, 'r') as f:
            motion = np.array(f['motion'])
            # TODO
            init_trans = np.mean(motion[0:26, :], axis=0)
            motion -= init_trans
            tmp_minmax = np.concatenate((np.amin(motion, axis=0)[:, None],
                                         np.amax(motion, axis=0)[:, None]), axis=1)
            # TODO
            tmp_minmax = tmp_minmax.T
        if 'pos_minmax' not in locals():
            pos_minmax = np.zeros((tmp_minmax.shape), dtype=np.float32)
            pos_minmax[0, :] = tmp_minmax[0, :]
            pos_minmax[1, :] = tmp_minmax[1, :]
        pos_minmax[0, :] = np.amin([tmp_minmax[0, :], pos_minmax[0, :]], axis=0)  # minimum
        pos_minmax[1, :] = np.amax([tmp_minmax[1, :], pos_minmax[1, :]], axis=0)  # maximum
        init += init_trans
    init /= len(fileslist)
    with h5py.File(configuration['file_pos_minmax'], 'a') as f:
        f.create_dataset('minmax', data=pos_minmax.T)
        f.create_dataset('init', data=init)


def main():
    type = 'train'
    data = exp
    prefix = os.path.join(data, type)
    if not os.path.exists(exp):
        os.makedirs(exp)
        os.makedirs(prefix)
    configuration[type] = prefix
    join = os.path.join(dataset_master, 'DANCE_*')
    folders = glob.glob(join)
    for i in range(len(folders)):
        path = folders[i]
        print('Using ', path)
        motion, start_position, end_position = output_loader(path)
        h5file = '{}f{:03d}.h5'.format(os.path.join(prefix, type), i)
        list_path = np.string_(path)
        with h5py.File(h5file, 'a') as f:
            f.create_dataset('song_path', data=list_path)
            f.create_dataset('motion', data=motion)
            f.create_dataset('position', data=[start_position, end_position])
        print('Making ', h5file)

    configuration['file_pos_minmax'] = os.path.join(data, 'pos_minmax.h5')

    if not os.path.exists(configuration['file_pos_minmax']):
        file_list = glob.glob(os.path.join(configuration[type], '*'))
        calculate_minmax(file_list)

    file_list = glob.glob(os.path.join(prefix, '*'))
    for i in range(len(file_list)):
        item = file_list[i]
        print('treatment of {}'.format(item))
        with h5py.File(item, 'a') as f:
            motion = np.array(f['motion'])
            pos = np.array(f['position'])
            start_pos = pos[0]
            end_pos = pos[1]
        motion, motion_mean = motion_transform(motion, configuration)
        motion = np.squeeze(motion)
        with h5py.File(item, 'a') as f:
            del f['motion']
            f.create_dataset('motion', data=motion)

    type = 'test'
    prefix = os.path.join(data, type)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    configuration[type] = prefix
    join = os.path.join(dataset_master, 'test', 'DANCE_*')
    folders = glob.glob(join)
    for i in range(len(folders)):
        path = folders[i]
        print('Using ', path)
        motion, start_position, end_position = output_loader(path)
        motion, motion_mean = motion_transform(motion, configuration)
        motion = np.squeeze(motion)
        h5file = '{}f{:03d}.h5'.format(os.path.join(prefix, type), i)
        list_path = np.string_(path)
        with h5py.File(h5file, 'a') as f:
            f.create_dataset('song_path', data=list_path)
            f.create_dataset('motion', data=motion)
            f.create_dataset('position', data=[start_position, end_position])
        print('Making ', h5file)


if __name__ == '__main__':
    main()
