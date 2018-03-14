import numpy as np
import nipype
from nipype.interfaces import niftyreg

from . import nifti
from . import constants as const
from .path import ensure_dir, get_temp_path

NEAREST = 'NN'
LINEAR = 'LIN'
CUBIV = 'CUB'
SINC = 'SINC'

nipype.config.set('logging', 'workflow_level', 'CRITICAL')
nipype.config.set('logging', 'interface_level', 'CRITICAL')
nipype.logging.update_logging(nipype.config)

def register(ref_path, flo_path, trsf_path=None, res_path=None,
             ref_mask_path=None, flo_mask_path=None, init_trsf_path=None,
             rigid_only=False, affine_directly=False):
    cleanup = []
    if trsf_path is None:
        trsf_path = get_temp_path('.txt')
        cleanup.append(trsf_path)
    if res_path is None:
        res_path = get_temp_path('.nii.gz')
        cleanup.append(res_path)

    aladin = niftyreg.RegAladin()
    aladin.inputs.ref_file = str(ref_path)
    aladin.inputs.flo_file = str(flo_path)
    aladin.inputs.aff_file = str(trsf_path)
    aladin.inputs.res_file = str(res_path)
    aladin.inputs.aff_direct_flag = affine_directly
    aladin.inputs.rig_only_flag = rigid_only

    if ref_mask_path is not None:
        aladin.inputs.rmask_file = str(ref_mask_path)
    if flo_mask_path is not None:
        aladin.inputs.fmask_file = str(flo_mask_path)
    if init_trsf_path is not None:
        aladin.inputs.in_aff_file = str(init_trsf_path)
    ensure_dir(res_path)
    ensure_dir(trsf_path)
    aladin.run()
    for path in cleanup:
        path.unlink()
    return aladin


def register_free_form(ref_path, flo_path, init_trsf_path,
                       trsf_path=None, res_path=None,
                       ref_mask_path=None, flo_mask_path=None,
                       bending_energy=const.BENDING_ENERGY_DEFAULT):
    cleanup = []
    if trsf_path is None:
        trsf_path = get_temp_path('.nii.gz')
        cleanup.append(trsf_path)
    if res_path is None:
        res_path = get_temp_path('.nii.gz')
        cleanup.append(res_path)

    reg_ff = niftyreg.RegF3D()
    reg_ff.inputs.ref_file = str(ref_path)
    reg_ff.inputs.flo_file = str(flo_path)
    reg_ff.inputs.cpp_file = str(trsf_path)
    reg_ff.inputs.res_file = str(res_path)
    reg_ff.inputs.be_val = bending_energy

    if ref_mask_path is not None:
        reg_ff.inputs.rmask_file = str(ref_mask_path)
    if flo_mask_path is not None:
        reg_ff.inputs.fmask_file = str(flo_mask_path)
    if init_trsf_path is not None:
        reg_ff.inputs.aff_file = str(init_trsf_path)
    ensure_dir(res_path)
    ensure_dir(trsf_path)
    reg_ff.run()
    for path in cleanup:
        path.unlink()
    return reg_ff


def resample(flo_path, ref_path, trsf_path, res_path, interpolation=SINC):
    node = niftyreg.RegResample()
    node.inputs.ref_file = str(ref_path)
    node.inputs.flo_file = str(flo_path)
    node.inputs.trans_file = str(trsf_path)
    node.inputs.out_file = str(res_path)
    node.inputs.inter_val = interpolation
    ensure_dir(res_path)
    node.run()
    return node


def spline_to_displacement_field(ref_path, input_path, output_path):
    node = niftyreg.RegTransform()
    node.inputs.ref1_file = str(ref_path)
    node.inputs.def_input = str(input_path)
    node.inputs.out_file = str(output_path)
    node.run()
    return node


def compute_mean_image(images_paths, output_path):
    first_nii = nifti.load(images_paths[0])
    data = first_nii.get_data().astype(float)
    for image_path in images_paths[1:]:
        nii = nifti.load(image_path)
        data += nii.get_data()
    data /= len(images_paths)
    nifti.save(data, first_nii.affine, output_path)


def compute_mean_labels(labels_paths, labels_paths_map):
    first_nii = nifti.load(labels_paths[0])
    data = first_nii.get_data()
    labels = np.unique(data)
    priors = []
    for label in labels:
        priors.append((data == label).astype(float))
    for labels_path in labels_paths[1:]:
        nii = nifti.load(labels_path)
        data = nii.get_data()
        for label in labels:
            priors[label] += (data == label).astype(float)
    for label in labels:
        priors[label] /= len(labels_paths)
        output_path = labels_paths_map[label]
        nifti.save(priors[label], first_nii.affine, output_path)
