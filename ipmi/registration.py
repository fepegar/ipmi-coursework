
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
    aladin.inputs.ref_file = str(ref_path)
    aladin.inputs.flo_file = str(flo_path)
    aladin.inputs.aff_file = str(trsf_path)
    aladin.inputs.res_file = str(res_path)
    aladin.run()
    return aladin


def resample(flo_path, ref_path, trsf_path, res_path, interpolation='SINC'):
    ensure_dir(res_path)
    node = niftyreg.RegResample()
    node.inputs.ref_file = str(ref_path)
    node.inputs.flo_file = str(flo_path)
    node.inputs.trans_file = str(trsf_path)
    node.inputs.out_file = str(res_path)
    node.inputs.inter_val = interpolation
    node.run()
    return node


def compute_mean_image(images_paths, output_path):
    return


def compute_mean_labels(labels_paths, labels_paths_map):
    return
