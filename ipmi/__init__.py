from . import path
from . import segmentation
from . import registration
from .subject import SegmentedSubject, UnsegmentedSubject


def get_segmented_subjects():
    dirs = path.segmented_subjects_dir.glob('?')
    subjects = [SegmentedSubject(d.name) for d in dirs]
    return subjects


def get_unsegmented_subjects():
    dirs = path.unsegmented_subjects_dir.glob('??')
    subjects = [UnsegmentedSubject(d.name) for d in dirs]
    return subjects
