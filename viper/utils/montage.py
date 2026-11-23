"""One image row per query: the query photograph followed by the retrieved scan images."""
import numpy as np

# BGR colours (OpenCV)
QUERY_BORDER = (255, 0, 0)       # blue
CORRECT_BORDER = (0, 200, 0)     # green
WRONG_BORDER = (0, 0, 255)       # red


def save_montage(query_image, retrieved_images, correct, out_path, height=240, border=5):
    """Write the montage as a jpg. Every tile is scaled to the same height. Unreadable images are skipped."""
    import cv2

    tiles = []
    colours = [QUERY_BORDER] + [CORRECT_BORDER if ok else WRONG_BORDER for ok in correct]

    for path, colour in zip([query_image] + list(retrieved_images), colours):
        image = cv2.imread(path)
        if image is None:
            continue

        image = cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))
        tiles.append(cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=colour))
        
    cv2.imwrite(out_path, np.hstack(tiles))
