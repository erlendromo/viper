import numpy as np
import cv2

def montage(q_img, db_imgs, out, h=240):
    tiles = []
    for p in [q_img] + db_imgs:
        im = cv2.imread(p); w = int(im.shape[1] * h / im.shape[0]); tiles.append(cv2.resize(im, (w, h)))
    
    tiles[0] = cv2.copyMakeBorder(tiles[0], 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 255))
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(out, np.hstack(tiles))
