import cv2
from tqdm import tqdm
from denseflow import DenseFlow

denseflow = DenseFlow(in_data='project.avi', in_type='video')

for x, y in tqdm(denseflow):
    cv2.imwrite('x_flow.jpg', x)
    cv2.imwrite('y_flow.jpg', y)