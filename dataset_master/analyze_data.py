import json, codecs
import time
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_filepaths(path_to_data='dataset-master'):
    data_path = dict()
    data_path['C'] = list()
    data_path['R'] = list()
    data_path['T'] = list()
    data_path['W'] = list()

    lst = os.listdir(path_to_data)
    lst.sort()
    print(lst)
    for folders in lst:
        if folders[:5] != 'DANCE':
            break
        if folders[:7] == 'DANCE_C':
            data_path['C'].append(path_to_data + "/" + folders)
        if folders[:7] == 'DANCE_R':
            data_path['R'].append(path_to_data + "/" + folders)
        if folders[:7] == 'DANCE_T':
            data_path['T'].append(path_to_data + "/" + folders)
        if folders[:7] == 'DANCE_W':
            data_path['W'].append(path_to_data + "/" + folders)

    return data_path


# Analyze one dance:
data_path = get_filepaths(path_to_data='.')

dance_c1 = data_path['C'][1]

config_path = dance_c1 + '/' + "config.json"
config = pd.read_json(config_path, typ='series')
skeletons_path = dance_c1 + '/' + 'skeletons.json'
with open(skeletons_path, 'r') as fin:
    data = np.array(json.load(fin)['skeletons'])

print('dance c1 config:\n', config)

max_vector = data.max(axis=0)
print(max_vector.shape)
