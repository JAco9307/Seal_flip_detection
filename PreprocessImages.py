import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class Bounds:
    max_h = 0
    max_w = 0
    min_h = 1080
    min_w = 1920
    Threshold = 50

    def __init__(self, bin_img):
        PointSet = np.argwhere(bin_img).tolist()
        Hset, Wset = map(list,zip(*PointSet))
        self.max_h = max(Hset)
        self.max_w = max(Wset)
        self.min_h = min(Hset)
        self.min_w = min(Wset)

    def get_img(self, img):
        size = max(self.min_h - self.max_h, self.min_w - self.max_w)
        center = [(self.min_h + self.max_h)/2, (self.min_w + self.max_w)/2]
        h2 = int(center[0] - size)
        h1 = max(0, int(center[0] + size))
        w2 = int(center[1] - size)
        w1 = int(center[1] + size)
        print(h1, h2, w1, w2)
        return img[h1:h2, w1:w2]


if __name__ == '__main__':

    paths = ["Test_white", "Train_white"]
    groups = ["Flipped", "NotFlipped"]

    for path in paths:
        for group in groups:
            i = 0
            loc = f"Training_w/{path}/{group}"
            for filename in os.listdir(loc):
                img = cv2.imread(os.path.join(loc, filename))
                if img is not None:
                    im_gray = cv2.imread(os.path.join(loc, filename), cv2.IMREAD_GRAYSCALE)
                    # im_bw = cv2.adaptiveThreshold(im_gray, 255,
                    #                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
                    im_bw = cv2.adaptiveThreshold(im_gray, 255,
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 99, 49)
                    #cv2.threshold(im_gray, 190, 255, cv2.THRESH_BINARY)[1]
                    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
                    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
                    mask = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, se1)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
                    b = Bounds(mask)
                    out = b.get_img(im_gray)
                    try:
                        cv2.imwrite(f'Training_p/{path}/{group}/{i}.png', out)
                    except:
                        print(f"Skipping {path}/{group} {filename}")
                    i += 1


