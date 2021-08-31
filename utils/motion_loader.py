import json
import numpy as np


def load_motions_features(dance_path):
    """
    :param dance_path: a string that give the path to the folder for one dance
    :return: a dictionary motions_features = {frame : list( motions features ) }
    """
    config_path = dance_path + '/' + "config.json"
    skeletons_path = dance_path + '/' + 'skeletons.json'

    with open(config_path) as fin:
        config = json.load(fin)
    with open(skeletons_path, 'r') as fin:
        motion_features = np.array(json.load(fin)['skeletons'])

    start_pos = config['start_position']

    X = motion_features.shape[0]
    nb_features = motion_features.shape[1] * motion_features.shape[2]
    motions_features = np.reshape(motion_features, (X, nb_features))
    end_pos = motions_features.shape[0] + start_pos

    return motions_features, start_pos, end_pos


def normalize_skeletons(data):
    data_min = np.amin(np.abs(data), axis=0)
    data_max = np.amax(np.abs(data), axis=0)
    one = np.ones(data.shape)
    normalize_data = (2 * (data - data_min) / (data_max - data_min)) - one
    return normalize_data, data_max, data_min


def output_loader(path_to_data='dataset_master'):
    motions_features, START_POS, END_POS = load_motions_features(path_to_data)

    return motions_features, START_POS, END_POS