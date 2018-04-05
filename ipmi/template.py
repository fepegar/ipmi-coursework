import shutil
from os.path import relpath

import numpy as np
from PIL import Image

from . import path
from . import nifti
from . import constants as const
from . import registration as reg


NII_EXT = '.nii.gz'
PNG_EXT = '.png'
PRIORS = '_priors'


def get_final_template():
    return Template(const.FINAL_TEMPLATE)



class Template:

    def __init__(self, template_id):
        self.id = template_id
        self.dir = path.templates_dir / self.id

        self.template_image_path = self.dir / (self.id + '_image' + NII_EXT)
        self.priors_background_path = self.dir / (
            self.id + PRIORS + '_background' + NII_EXT)
        self.priors_csf_path = self.dir / (
            self.id + PRIORS + '_csf' + NII_EXT)
        self.priors_gm_path = self.dir / (
            self.id + PRIORS + '_gm' + NII_EXT)
        self.priors_wm_path = self.dir / (
            self.id + PRIORS + '_wm' + NII_EXT)
        self.priors_paths_map = {
            const.BACKGROUND: self.priors_background_path,
            const.CSF: self.priors_csf_path,
            const.GREY_MATTER: self.priors_gm_path,
            const.WHITE_MATTER: self.priors_wm_path,
        }
        self.collage_path = self.dir / (self.id + '_collage' + PNG_EXT)
        self.std_path = self.dir / (self.id + '_std' + NII_EXT)


    def exists(self):
        return self.template_image_path.is_file()


    def make_default(self):
        template_final = Template(const.FINAL_TEMPLATE)
        if template_final.dir.is_dir():
            shutil.rmtree(str(template_final.dir))
        path.ensure_dir(template_final.template_image_path)

        start = template_final.dir

        template_final.template_image_path.symlink_to(
            relpath(self.template_image_path, start=start))
        template_final.priors_background_path.symlink_to(
            relpath(self.priors_background_path, start=start))
        template_final.priors_csf_path.symlink_to(
            relpath(self.priors_csf_path, start=start))
        template_final.priors_gm_path.symlink_to(
            relpath(self.priors_gm_path, start=start))
        template_final.priors_wm_path.symlink_to(
            relpath(self.priors_wm_path, start=start))
        template_final.collage_path.symlink_to(
            relpath(self.collage_path, start=start))


    def generate(self, images_paths, labels_paths):
        reg.compute_mean_image(images_paths, self.template_image_path)
        reg.compute_mean_labels(labels_paths, self.priors_paths_map)


    def make_collage_all(self, output_path=None, force=False):
        if output_path is None:
            output_path = self.collage_path

        if output_path.is_file() and not force:
            return

        images_paths = [
            self.template_image_path,
            self.priors_csf_path,
            self.priors_gm_path,
            self.priors_wm_path,
        ]
        images = [self.get_collage(nii_path) for nii_path in images_paths]
        top = np.hstack(images[:2])
        bottom = np.hstack(images[2:])
        result = np.vstack([top, bottom])
        image = Image.fromarray(result).convert('RGB')
        image.save(str(output_path))


    def get_collage(self, nii_path):
        """Template images are in LIA orientation"""

        def normalise_int(data):
            data = data.astype(float)
            data -= data.min()
            data /= data.max()
            data *= 255
            data = data.astype(np.uint8)
            return data

        nii = nifti.load(nii_path)
        data = nii.get_data()
        data = normalise_int(data)

        sagittal_slice = data[100, :, :]
        sagittal_slice = np.fliplr(sagittal_slice)

        axial_slice = data[:, 80, :]
        axial_slice = np.rot90(axial_slice)

        coronal_slice = data[:, :, 115]
        coronal_slice = np.rot90(coronal_slice, -1)
        coronal_slice = np.fliplr(coronal_slice)

        sy, _ = axial_slice.shape
        gap = np.zeros([sy, sy])

        top = np.hstack([gap, axial_slice])
        bottom = np.hstack([sagittal_slice, coronal_slice])
        result = np.vstack([top, bottom])

        return result


    def make_std_image(self):
        from . import get_segmented_subjects
        subjects = get_segmented_subjects()
        paths = [s.get_image_on_template_path(self) for s in subjects]
        first = nifti.load(paths[0])
        images = np.empty(list(first.shape) + [len(paths)])

        for i, path in enumerate(paths):
            images[..., i] = nifti.load(path).get_data()

        result = images.std(axis=3)
        nifti.save(result, first.affine, self.std_path)
