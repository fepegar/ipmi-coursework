import os
from pathlib import Path
from subprocess import call

import numpy as np
import nibabel as nib

import ipmi
from ipmi import template
from ipmi import registration as reg

os.chdir('/tmp/ff')


def change_range(data, minimum, maximum):
    data -= data.min()
    data /= data.max()
    data *= maximum - minimum
    data += minimum
    data = data.astype(np.uint16)
    return data


def make_label_map(input_path, central_value, output_path):
    nii = nib.load(input_path)
    data = nii.get_data()
    where_nan = np.isnan(data)
    np.nan_to_num(data, copy=False)

    if central_value - data.min() > data.max() - central_value:
        m = (128 - 0) / (central_value - data.min())
        n = 128 - m * central_value
        min_label = 0
        max_label = int(m * data.max() + n)
    else:
        m = (255 - 128) / (data.max() - central_value)
        n = 128 - m * central_value
        min_label = int(m * data.min() + n)
        max_label = 256
    labels = change_range(data, min_label, max_label)
    labels[where_nan] = 1000
    nii = nib.Nifti1Image(labels, nii.affine)
    nib.save(nii, output_path)


def main():
    subject = ipmi.get_unsegmented_subjects()[0]
    ref_path = subject.t1_path
    flo_path = Path('template_on_00_affine.nii.gz')
    if not flo_path.is_file():
        template_path = template.get_final_template().template_image_path
        trsf_path = subject.template_to_t1_affine_path
        reg.resample(template_path, ref_path, trsf_path, flo_path)

    bending_energies = np.logspace(-3, -1, 10)

    for be in bending_energies:
        # Register
        command = (f'reg_f3d -ref {ref_path} -flo {flo_path} '
                   f'-cpp cpp_be_{be:.5f}.nii.gz '
                   f'-res res_be_{be:.5f}.nii.gz -be {be}'.split())
        call(command)

        # Jacobian
        command = (f'reg_jacobian -ref {ref_path} '
                   f'-trans cpp_be_{be:.5f}.nii.gz -jac jac_be_{be:.5f}.nii.gz '
                   f'-jacL jacL_be_{be:.5f}.nii.gz'.split())
        call(command)

        # Jacobian label maps for 3D Slicer
        make_label_map(f'jacL_be_{be:.5f}.nii.gz', 0,
                       f'jacL_be_{be:.5f}_label_map.nii.gz')
        make_label_map(f'jac_be_{be:.5f}.nii.gz', 1,
                       f'jac_be_{be:.5f}_label_map.nii.gz')

        reg.spline_to_displacement_field(f'{ref_path}',
                                         f'cpp_be_{be:.5f}.nii.gz',
                                         f'cpp_be_{be:.5f}_disp.nii.gz')


if __name__ == '__main__':
    main()
