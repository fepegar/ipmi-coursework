import numpy as np
import nibabel as nib


def load(path):
    nii = nib.load(str(path))
    return nii


def save(data, affine, path):
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(path))


def get_voxel_volume(nii):
    dims = nii.header['pixdim'][1:4]
    voxel_volume = np.prod(dims)
    return voxel_volume
