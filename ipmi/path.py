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
subjects_dir = output_dir / 'subjects'
templates_dir = output_dir / 'templates'
segmented_subjects_dir = subjects_dir / 'segmented'
unsegmented_subjects_dir = subjects_dir / 'unsegmented'

templates_gif_path = templates_dir / 'template_evolution.gif'
templates_gif_reflect_path = templates_dir / 'template_evolution_reflect.gif'
dice_report_path = output_dir / 'dice_scores.csv'
dice_report_pandas_path = output_dir / 'dice_scores_pandas.csv'
correlations_report_path = output_dir / 'correlations.csv'


def ensure_dir(path):
    """Make sure that the directory and its parents exists"""
    path = Path(path)
    if path.exists():
        return
    is_dir = not path.suffixes
    if is_dir:
        path.mkdir(parents=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


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
