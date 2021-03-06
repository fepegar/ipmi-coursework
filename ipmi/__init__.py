from . import path
from . import nifti
from . import segmentation
from . import registration
from .template import Template
from .subject import SegmentedSubject, UnsegmentedSubject


def get_segmented_subjects():
    dirs = sorted(path.segmented_subjects_dir.glob('?'))
    subjects = [SegmentedSubject(d.name) for d in dirs]
    return subjects


def get_unsegmented_subjects():
    dirs = sorted(path.unsegmented_subjects_dir.glob('??'))
    subjects = [UnsegmentedSubject(d.name) for d in dirs]
    return subjects


def get_all_subjects():
    return get_segmented_subjects() + get_unsegmented_subjects()


def get_subject_by_id(subjects, subject_id):
    for subject in subjects:
        if subject.id == subject_id:
            found = subject
            break
    else:
        found = None
    return found
