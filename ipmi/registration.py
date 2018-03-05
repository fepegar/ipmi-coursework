import tempfile

from nipype.interfaces import niftyreg

NN = 'NN'
LIN = 'LIN'
CUB = 'CUB'
SINC = 'SINC'

def register(ref_path, flo_path, trsf_path=None, res_path=None):
    return


def resample(flo_path, ref_path, trsf_path, res_path, interpolation='SINC'):
    return


def compute_mean_image(images_paths, output_path):
    return


def compute_mean_labels(labels_paths, labels_paths_map):
    return
