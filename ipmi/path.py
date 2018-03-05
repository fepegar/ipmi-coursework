import re

from . import constants as const

segmented_images_dir = const.ROOT_DIR / 'Segmented_images'
unsegmented_images_dir = const.ROOT_DIR / 'Unsegmented_images'
output_dir = const.ROOT_DIR / 'output'
registration_dir = output_dir / 'registration'
template_path = registration_dir / 'template.nii.gz'
priors_csf_path = registration_dir / 'csf_priors.nii.gz'
priors_gm_path = registration_dir / 'gm_priors.nii.gz'
priors_wm_path = registration_dir / 'wm_priors.nii.gz'


def ensure_dir(path):
    if path.exists():
        return
    is_dir = not path.suffixes
    if is_dir:
        path.mkdir(parents=True)
    else:
        path.parents[0].mkdir(parents=True)

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
