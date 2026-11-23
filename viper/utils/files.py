"""Finding images on disk and reading the frame id from a file name."""
import glob
import os
import re

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg")


def file_id(path):
    """The last number in the file name, e.g. 'frame_00012.png' -> 12. None if the name has no number.

    Scan images and pose files are paired by this id, and the gt_image_ids in the IHM json refer to it.
    """
    numbers = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    
    return int(numbers[-1]) if numbers else None


def list_images(root):
    """All images below root (any depth), sorted by path."""
    return sorted(
        path
        for ext in IMAGE_EXTENSIONS
        for path in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    )
