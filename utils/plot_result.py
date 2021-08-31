import numpy as np
import os
import cv2
import json
import logging
import h5py
from moviepy.editor import *

logger = logging.getLogger(__name__)

from motion_transform import reverse_motion_transform

CANVAS_SIZE = (400, 600, 3)
videoWriter = None


def draw(frames, export_to_file=False, videoWriter_enable=None):
    'Draw whole video, frames by frame.'
    global CANVAS_SIZE
    global videoWriter
    if videoWriter_enable is not None:
        videoWriter = videoWriter_enable
    frames[:, :, 0] += CANVAS_SIZE[0] // 2
    frames[:, :, 1] += CANVAS_SIZE[1] // 2
    for i in range(len(frames)):
        cvs = np.ones(CANVAS_SIZE)
        color = (0, 0, 0)
        hlcolor = (255, 0, 0)
        dlcolor = (0, 0, 255)
        for points in frames[i]:
            cv2.circle(cvs, (int(points[0]), int(points[1])), radius=4, thickness=-1, color=hlcolor)
        frame = frames[i]
        cv2.line(cvs, (int(frame[0][0]), int(frame[0][1])), (int(frame[1][0]), int(frame[1][1])), color, 2)
        cv2.line(cvs, (int((frame[0][0] + frame[1][0]) / 2), int((frame[0][1] + frame[1][1]) / 2)),
                 (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), color, 2)
        cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])),
                 (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), color, 2)
        cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[4][0]), int(frame[4][1])), color, 2)
        cv2.line(cvs, (int(frame[4][0]), int(frame[4][1])), (int(frame[5][0]), int(frame[5][1])), color, 2)
        cv2.line(cvs, (int(frame[5][0]), int(frame[5][1])), (int(frame[6][0]), int(frame[6][1])), color, 2)
        cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])),
                 (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), color, 2)
        cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[13][0]), int(frame[13][1])), color, 2)
        cv2.line(cvs, (int(frame[13][0]), int(frame[13][1])), (int(frame[14][0]), int(frame[14][1])), color, 2)
        cv2.line(cvs, (int(frame[14][0]), int(frame[14][1])), (int(frame[15][0]), int(frame[15][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])),
                 (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[7][0]), int(frame[7][1])), color, 2)
        cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[8][0]), int(frame[8][1])), color, 2)
        cv2.line(cvs, (int(frame[8][0]), int(frame[8][1])), (int(frame[9][0]), int(frame[9][1])), color, 2)
        cv2.line(cvs, (int(frame[9][0]), int(frame[9][1])),
                 (int((frame[10][0] + frame[11][0]) / 2), int((frame[10][1] + frame[11][1]) / 2)), color, 2)
        cv2.line(cvs, (int(frame[10][0]), int(frame[10][1])), (int(frame[11][0]), int(frame[11][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)
        cv2.line(cvs, (int(frame[16][0]), int(frame[16][1])), (int(frame[17][0]), int(frame[17][1])), color, 2)
        cv2.line(cvs, (int(frame[17][0]), int(frame[17][1])), (int(frame[18][0]), int(frame[18][1])), color, 2)
        cv2.line(cvs, (int(frame[18][0]), int(frame[18][1])),
                 (int((frame[19][0] + frame[20][0]) / 2), int((frame[19][1] + frame[20][1]) / 2)), color, 2)
        cv2.line(cvs, (int(frame[19][0]), int(frame[19][1])), (int(frame[20][0]), int(frame[20][1])), color, 2)

        '''
        for j in range(23):
            cv2.putText(cvs,str(j),(int(frame[j][0]),int(frame[j][1])),cv2.FONT_HERSHEY_SIMPLEX,.4, (155, 0, 255), 1, False)
            '''
        if export_to_file:
            img8 = (np.flip(cvs, 0) * (2 ** 8 - 1)).astype(np.uint8)
            videoWriter.write(img8)
        else:
            cv2.imshow('canvas', np.flip(cvs, 0))
            cv2.waitKey(0)
    pass


def test_draw(name, configuration, speed=25, nb_examples=50, export_to_file=False, test=False, untransformed=False):
    if test:
        name = 'dataset_master/DANCE_W_22'
        with open('../%s/config.json' % name) as fin:
            config = json.load(fin)
        print(config)
        with open('../%s/skeletons.json' % name, 'r') as fin:
            data = np.array(json.load(fin)['skeletons'])

        data = data[0:nb_examples]
        name = name[name.find('/') + 1:]
    else:
        item = name
        with h5py.File(item, 'r') as f:
            name = str(np.array(f['song_path']))[1:]
            data = np.array(f['motion'])
            config = np.array(f['position'])
        name = name[name.rfind('/'):-1]
        if not untransformed:
            data = reverse_motion_transform(data, configuration)
        data = np.reshape(data, (data.shape[0], 23, 3))
        data = data[0:nb_examples]
        print(name)

    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # opencv3.0
    videoWriter = cv2.VideoWriter('./' + name + '.avi', fourcc, speed, (600, 400))
    draw(data, export_to_file=export_to_file)
    videoWriter.release()
    cv2.destroyAllWindows()

    os.system('ffmpeg -i ./%s.avi ./%s.mp4' % (name, name))
    os.system('rm ./%s.avi' % name)

    if export_to_file:
        movie_dance = VideoFileClip('./%s.mp4' % name, audio=False)
        movie_dance.write_videofile("./%s.avi" % name, fps=25, codec='libx264')

        os.system('rm ./%s.mp4' % name)
        os.system('ffmpeg -i ./%s.avi ./%s.mp4' % (name, name))
        os.system('rm ./%s.avi' % name)
    else:
        os.system('rm ./%s.mp4' % name)

    logger.info('Finish <%s>' % name)


def draw_image(frame):
    'Draw only one frame'
    frame[:, 0] += CANVAS_SIZE[0] // 2
    frame[:, 1] += CANVAS_SIZE[1] // 2
    cvs = np.ones(CANVAS_SIZE)
    color = (0, 0, 0)
    hlcolor = (255, 0, 0)
    dlcolor = (0, 0, 255)
    for points in frame:
        cvs = cv2.circle(cvs, (int(points[0]), int(points[1])), radius=4, thickness=-1, color=hlcolor)
    cvs = cv2.line(cvs, (int(frame[0][0]), int(frame[0][1])), (int(frame[1][0]), int(frame[1][1])), color, 2)
    cvs = cv2.line(cvs, (int((frame[0][0] + frame[1][0]) / 2), int((frame[0][1] + frame[1][1]) / 2)),
                   (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), color, 2)
    cvs = cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])),
                   (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), color, 2)
    cvs = cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[4][0]), int(frame[4][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[4][0]), int(frame[4][1])), (int(frame[5][0]), int(frame[5][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[5][0]), int(frame[5][1])), (int(frame[6][0]), int(frame[6][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])),
                   (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), color, 2)
    cvs = cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[13][0]), int(frame[13][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[13][0]), int(frame[13][1])), (int(frame[14][0]), int(frame[14][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[14][0]), int(frame[14][1])), (int(frame[15][0]), int(frame[15][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])),
                   (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), color, 2)
    cvs = cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[7][0]), int(frame[7][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[8][0]), int(frame[8][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[8][0]), int(frame[8][1])), (int(frame[9][0]), int(frame[9][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[9][0]), int(frame[9][1])),
                   (int((frame[10][0] + frame[11][0]) / 2), int((frame[10][1] + frame[11][1]) / 2)), color, 2)
    cvs = cv2.line(cvs, (int(frame[10][0]), int(frame[10][1])), (int(frame[11][0]), int(frame[11][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[16][0]), int(frame[16][1])), (int(frame[17][0]), int(frame[17][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[17][0]), int(frame[17][1])), (int(frame[18][0]), int(frame[18][1])), color, 2)
    cvs = cv2.line(cvs, (int(frame[18][0]), int(frame[18][1])),
                   (int((frame[19][0] + frame[20][0]) / 2), int((frame[19][1] + frame[20][1]) / 2)), color, 2)
    cvs = cv2.line(cvs, (int(frame[19][0]), int(frame[19][1])), (int(frame[20][0]), int(frame[20][1])), color, 2)
    return cvs


if __name__ == '__main__':
    speed = 25
    nb_examples = 50
    export_to_file = False
    configuration = {'file_pos_minmax': './data/pos_minmax.h5',
                     'normalisation': 'interval',
                     'rng_pos': [-0.9, 0.9]}

    test = False
    if test:
        name = 'dataset_master/DANCE_W_22'
        with open('../%s/config.json' % name) as fin:
            config = json.load(fin)
        print(config)
        with open('../%s/skeletons.json' % name, 'r') as fin:
            data = np.array(json.load(fin)['skeletons'])

        data = data[0:nb_examples]
        name = name[name.find('/') + 1:]
    else:
        item = './data/train/trainf000.h5'
        with h5py.File(item, 'r') as f:
            name = str(np.array(f['song_path']))[1:]
            data = np.array(f['motion'])
            config = np.array(f['position'])
        name = name[name.rfind('/') + 1:-1]
        data = reverse_motion_transform(data, configuration)
        data = np.reshape(data, (data.shape[0], 23, 3))
        data = data[0:nb_examples]
        print(name)

    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # opencv3.0
    videoWriter = cv2.VideoWriter('./' + name + '.avi', fourcc, speed, (600, 400))
    draw(data, export_to_file=export_to_file)
    videoWriter.release()
    cv2.destroyAllWindows()

    os.system('ffmpeg -i ./%s.avi ./%s.mp4' % (name, name))
    os.system('rm ./%s.avi' % name)

    if export_to_file:
        movie_dance = VideoFileClip('./%s.mp4' % name, audio=False)
        movie_dance.write_videofile("./%s.avi" % name, fps=25, codec='libx264')

        os.system('rm ./%s.mp4' % name)
        os.system('ffmpeg -i ./%s.avi ./%s.mp4' % (name, name))
        os.system('rm ./%s.avi' % name)
    else:
        os.system('rm ./%s.mp4' % name)

    logger.info('Finish <%s>' % name)
