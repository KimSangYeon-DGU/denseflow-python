'''DenseFlow
Author: Sangyeon Kim
Description: This class is based on opencv's optical flow implementation
    and https://github.com/yjxiong/dense_flow
Prerequisite: OpenCV, OpenCV-contrib, numpy
'''
import cv2
import os
import numpy as np


class DenseFlow():
    def __init__(self, in_data, in_type,
                 resize=(224, 224),
                 out_dir='./data',
                 extractor='TVL1'):
        '''
            in_data: Input data to extract optical flows.
            in_type: Type of input data, e.g. video or directory that has images.
            out_dir: Directory for optical flow image to be saved.
        '''

        self.in_data = in_data
        self.in_type = in_type
        self.out_dir = out_dir
        self.extractor = cv2.optflow.DualTVL1OpticalFlow_create()
        self.resize = resize
        self.frames = []
        self.index = 1
        self.prev_frame = None
        self.curr_frame = None
        self._preprocess()

    def _preprocess(self):
        '''Extracts frames depending on type of input data.
        '''
        if self.in_type == 'dir':
            filenames = os.listdir(self.in_data)
            filenames.sort()
            for filename in filenames:
                temp_img = cv2.imread('/'.join((self.in_data, filename)))
                self.frames.append(
                    cv2.resize(temp_img, self.resize)
                )
        elif self.in_type == 'video':
            cap = cv2.VideoCapture(self.in_data)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break

                self.frames.append(cv2.resize(frame, self.resize))
        else:
            raise Exception('This type is not applicable.')

        if len(self.frames) <= 1:
            raise Exception('The number of images should be greater than 1.')

        print('Total frames: {0}'.format(len(self.frames)))

    def _save(self, data):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def _cast(self, img, lower_bound, upper_bound):
        _img = img
        _img[_img > upper_bound] = 255
        _img[_img < lower_bound] = 0
        _img[(lower_bound <= _img) & (_img <= upper_bound)] = np.round(
            255*((_img[(lower_bound <= _img) & (_img <= upper_bound)]) - (lower_bound))/((upper_bound)-(lower_bound)))
        return _img

    def _postprocess(self, x, y, lower_bound, upper_bound):
        return self._cast(x, lower_bound, upper_bound).astype('uint8'), self._cast(y, lower_bound, upper_bound).astype('uint8')

    def __next__(self):
        if self.index < len(self.frames):
            self.prev_frame = self.frames[self.index - 1]
            self.curr_frame = self.frames[self.index]
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(self.curr_frame, cv2.COLOR_BGR2GRAY)
            flow = self.extractor.calc(prev_gray, curr_gray, None)

            x = flow[..., 0]
            y = flow[..., 1]
            x, y = self._postprocess(x, y, -15, 15)
            x = x.astype('uint8')
            y = y.astype('uint8')
            self.index += 1
            return self.curr_frame, x, y
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frames) - 1
