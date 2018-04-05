import numpy as np
import nibabel as nib

from .path import ensure_dir

def load(path):
    nii = nib.load(str(path))
    return nii


def save(data, affine, path, settings=None):
    nii = nib.Nifti1Image(data, affine)
    if settings is not None:
        for method, value in settings.items():
            getattr(nii.header, method)(value)
    ensure_dir(path)
    nib.save(nii, str(path))


def get_voxel_volume(nii):
    dims = nii.header['pixdim'][1:4]
    voxel_volume = np.prod(dims)
    return voxel_volume


def get_thumbnail(image):

    def normalise_int(data):
        data = data.astype(float)
        data -= data.min()
        data /= data.max()
        data *= 255
        data = data.astype(np.uint8)
        return data

    if not isinstance(image, nib.Nifti1Image):
        if isinstance(image, np.ndarray):
            data = image
        else:  # assume PathLike
            image = load(image)
            data = image.get_data()

    # Extract slices from 3D array
    sagittal_slice = data[100, :, :]
    sagittal_slice = np.fliplr(sagittal_slice).squeeze()

    axial_slice = data[:, 80, :]
    axial_slice = np.rot90(axial_slice).squeeze()

    coronal_slice = data[:, :, 115]
    coronal_slice = np.rot90(coronal_slice, -1)
    coronal_slice = np.fliplr(coronal_slice).squeeze()

    # Grey to RGB
    if sagittal_slice.ndim == 2:
        sagittal_slice = np.stack(3 * [sagittal_slice], axis=2)
    if axial_slice.ndim == 2:
        axial_slice = np.stack(3 * [axial_slice], axis=2)
    if coronal_slice.ndim == 2:
        coronal_slice = np.stack(3 * [coronal_slice], axis=2)

    # Pad if odd and match width of sagittal slice
    if sagittal_slice.shape[1] % 2 == 1:
        width = [(0, 0), (0, 1), (0, 0)]
        sagittal_slice = np.pad(sagittal_slice, width, 'constant')

    if axial_slice.shape[1] % 2 == 1:
        width = [(0, 0), (0, 1), (0, 0)]
        axial_slice = np.pad(axial_slice, width, 'constant')
    diff = sagittal_slice.shape[1] - axial_slice.shape[1]
    width = [(0, 0), (diff//2, diff//2), (0, 0)]
    axial_slice = np.pad(axial_slice, width, 'constant')

    if coronal_slice.shape[1] % 2 == 1:
        width = [(0, 0), (0, 1), (0, 0)]
        coronal_slice = np.pad(coronal_slice, width, 'constant')
    diff = sagittal_slice.shape[1] - coronal_slice.shape[1]
    width = [(0, 0), (diff//2, diff//2), (0, 0)]
    coronal_slice = np.pad(coronal_slice, width, 'constant')

    thumbnail = np.vstack([axial_slice, sagittal_slice, coronal_slice])

    return normalise_int(thumbnail)
