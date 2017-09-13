from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
import time
import argparse
import pyflow
import os
from image_processor import process_image

# Flow Options:
alpha = 0.012
ratio = 0.5625
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

parser = argparse.ArgumentParser(
    description='Python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

if __name__ == '__main__':
    root_dir = 'data/jpg_320_180_1fps/'
    output_dir = 'data/jpg_320_180_1fps_OF/'
    exclude_prefixes = ('__', '.')  # exclusion prefixes
    complete_paths = []
    file_names = []
    filename = -1
    counter = 0
    ims = []
    print("hej")
    for dirpath, dirnames, files in os.walk(root_dir):
        if '.DS_Store' not in files[0]:
            for filename in files:
                print(filename)
                print(dirpath)
                im = process_image(dirpath + '/' + filename, (320, 180, 3))
                ims.append(im)
                counter += 1
                if counter == 2:
                    break
            im1, im2 = ims[0], ims[1]
            im1 = im1.astype(float) / 255.
            im2 = im2.astype(float) / 255.
            s = time.time()
            u, v, im2W = pyflow.coarse2fine_flow(
                        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                        nSORIterations, colType)
            e = time.time()
            print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
                e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)
            # import ipdb; ipdb.set_trace()
            np.save(output_dir + 'examples/outFlow.npy', flow)

            if args.viz:
                import cv2

                hsv = np.zeros(im1.shape, dtype=np.uint8)
                hsv[:, :, 0] = 255
                hsv[:, :, 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                # mag, ang = cart2pol(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(output_dir + 'examples/outFlow_new.png', rgb)
                cv2.imwrite(output_dir + 'examples/car2Warped_new.jpg', im2W[:, :, ::-1] * 255)

            break