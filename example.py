import cv2
from tqdm import tqdm
from denseflow import DenseFlow

denseflow = DenseFlow(in_data='project.avi', in_type='video')

idx = 0
for img, x, y in tqdm(denseflow):
    cv2.imwrite('./images/img/img{0:03d}.png'.format(idx), img)
    cv2.imwrite('./images/x_flow/x_flow_{0:03d}.png'.format(idx), x)
    cv2.imwrite('./images/y_flow/y_flow_{0:03d}.png'.format(idx), y)
    idx+=1
