import nibabel as nib


def load(path):
    nii = nib.load(str(path))
    return nii


def save(data, affine, path):
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, str(path))
