import pandas as pd
import numpy as np
import cv2

from image_processor import process_image


def visualize(im, flow):
    import ipdb;
    ipdb.set_trace()
    hsv = np.zeros(im.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('examples/ex.png', rgb)

df = pd.read_csv('data/jpg_320_180_1fps_OF/horse_4.csv')
row_idx = 53
row = df.loc[53]

rgb = row['Path']
of = row['OF_Path']



im = process_image(rgb, (320,180,3))
flow = np.load(of)

visualize(im, flow)



