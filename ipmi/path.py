import re

import constants as const

segmented_images_dir = const.ROOT_DIR / 'Segmented_images'
unsegmented_images_dir = const.ROOT_DIR / 'Unsegmented_images'

def get_unsegmented_images_and_ages():
    images_paths = sorted(unsegmented_images_dir.glob('*.nii.gz'))
    stems = [path.stem for path in images_paths]
    pattern = r'.*_(\d+\.?\d?).*'
    ages = [float(re.match(pattern, stem).groups()[0]) for stem in stems]
    return images_paths, ages

def get_segmented_images_and_labels():
    images_paths = sorted(segmented_images_dir.glob('*img.nii.gz'))
    labels_paths = sorted(segmented_images_dir.glob('*seg.nii.gz'))
    return images_paths, labels_paths

if __name__ == '__main__':
    print(get_segmented_images_and_labels())
