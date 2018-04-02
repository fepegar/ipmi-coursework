import os
from subprocess import call

import numpy as np
import nibabel as nib

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
    data[where_nan] = 0  # remove NaNs

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
    for be in np.logspace(-3, -1, 10):
        # Register
        command = ('reg_f3d -ref 00_t1.nii.gz -flo final_image_on_00.nii.gz '
                   f'-cpp cpp_be_{be:.5f}.nii.gz '
                   f'-res res_be_{be:.5f}.nii.gz -be {be}'.split())
        call(command)

        # Jacobian
        command = ('reg_jacobian -ref 00_t1.nii.gz '
                   f'-trans cpp_be_{be:.5f}.nii.gz -jac jac_be_{be:.5f}.nii.gz '
                   f'-jacL jacL_be_{be:.5f}.nii.gz'.split())
        call(command)

        # Jacobian label maps for 3D Slicer
        make_label_map(f'jacL_be_{be:.5f}.nii.gz', 0,
                       f'jacL_be_{be:.5f}_label_map.nii.gz')
        make_label_map(f'jac_be_{be:.5f}.nii.gz', 1,
                       f'jac_be_{be:.5f}_label_map.nii.gz')

        reg.spline_to_displacement_field('00_t1.nii.gz',
                                         f'cpp_be_{be:.5f}.nii.gz',
                                         f'cpp_be_{be:.5f}_disp.nii.gz')


if __name__ == '__main__':
    main()
