
from nipype.interfaces import niftyreg

from .path import ensure_dir, get_temp_path

NN = 'NN'
LIN = 'LIN'
CUB = 'CUB'
SINC = 'SINC'

def register(ref_path, flo_path, trsf_path=None, res_path=None):
    if trsf_path is None:
        trsf_path = get_temp_path('.txt')
    if res_path is None:
        res_path = get_temp_path('.nii.gz')
    ensure_dir(res_path)
    ensure_dir(trsf_path)
    aladin = niftyreg.RegAladin()
    aladin.inputs.ref_file = ref_path
    aladin.inputs.flo_file = flo_path
    aladin.inputs.aff_file = trsf_path
    aladin.inputs.res_file = res_path
    aladin.run()
    return aladin


def resample(flo_path, ref_path, trsf_path, res_path, interpolation='SINC'):
    return


def compute_mean_image(images_paths, output_path):
    return


def compute_mean_labels(labels_paths, labels_paths_map):
    return
