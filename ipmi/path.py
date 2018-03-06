"""
Module for handling project paths
"""
import re
from pathlib import Path
from random import choice
from tempfile import gettempdir
from string import digits, ascii_letters

from . import constants as const

segmented_images_dir = const.ROOT_DIR / 'Segmented_images'
unsegmented_images_dir = const.ROOT_DIR / 'Unsegmented_images'
output_dir = const.ROOT_DIR / 'output'
registration_dir = output_dir / 'registration'
transforms_dir = registration_dir / 'transforms'
resampled_images_dir = registration_dir / 'resampled_images'
resampled_labels_dir = registration_dir / 'resampled_labels'
resampled_priors_dir = registration_dir / 'resampled_priors'
priors_dir = registration_dir / 'priors'
transforms_groupwise_dir = transforms_dir / 'groupwise'
transforms_from_template_dir = transforms_dir / 'from_template'
priors_background_dir = resampled_priors_dir / 'background'
priors_csf_dir = resampled_priors_dir / 'csf'
priors_gm_dir = resampled_priors_dir / 'gm'
priors_wm_dir = resampled_priors_dir / 'wm'

template_path = registration_dir / 'template.nii.gz'
priors_background_path = priors_dir / 'background_priors.nii.gz'
priors_csf_path = priors_dir / 'csf_priors.nii.gz'
priors_gm_path = priors_dir / 'gm_priors.nii.gz'
priors_wm_path = priors_dir / 'wm_priors.nii.gz'
priors_map = {
    0: priors_background_path,
    1: priors_csf_path,
    2: priors_gm_path,
    3: priors_wm_path,
}


def ensure_dir(path):
    """Make sure that the directory and its parents exists"""
    if path.exists():
        return
    is_dir = not path.suffixes
    if is_dir:
        path.mkdir(parents=True)
    else:
        path.parents[0].mkdir(parents=True, exist_ok=True)


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


def get_temp_path(suffix):
    random_str = ''.join([choice(ascii_letters + digits) for n in range(32)])
    return Path(gettempdir(), random_str).with_suffix(suffix)
