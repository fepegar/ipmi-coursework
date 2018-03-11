import re

from . import path
from . import constants as const
from . import segmentation as seg
from . import registration as reg
from .template import get_final_template

NII_EXT = '.nii.gz'
TXT_EXT = '.txt'
AFFINE_EXT = TXT_EXT

T1 = '_t1'
LABEL_MAP = '_label_map'
AGE = '_age'
SEGMENTATION = '_segmentation'
PRIORS = '_priors'


class Subject:
    #pylint: disable=E1101

    def __init__(self, subject_id):
        self.id = subject_id


    def __repr__(self):
        return f'Subject {self.id}'


    def get_t1_path(self):
        return self.dir / (self.id + T1 + NII_EXT)


    def import_t1(self, t1_path):
        if not self.t1_path.is_file():
            path.ensure_dir(self.t1_path)
            self.t1_path.symlink_to(t1_path)


    def get_affine_to_template_path(self, template):
        aff_path = self.transforms_dir / (
            self.id + T1 + f'_to_{template.id}' + AFFINE_EXT)
        return aff_path


    def get_image_on_template_path(self, template):
        res_path = self.resampled_dir / (
            self.id + T1 + f'_on_{template.id}' + NII_EXT)
        return res_path


    def get_label_map_on_template_path(self, template):
        res_path = self.resampled_dir / (
            self.id + LABEL_MAP + f'_on_{template.id}' + NII_EXT)
        return res_path



class SegmentedSubject(Subject):

    def __init__(self, subject_id, t1_path=None, label_map_path=None):
        super().__init__(subject_id)
        self.dir = path.segmented_subjects_dir / self.id
        self.transforms_dir = self.dir / 'transforms'
        self.resampled_dir = self.dir / 'resampled'

        self.t1_path = self.get_t1_path()
        self.label_map_path = self.dir / (self.id + LABEL_MAP + NII_EXT)
        self.brain_mask_path = self.dir / (self.id + '_brain_mask' + NII_EXT)
        self.t1_masked_path = self.dir / (self.id + T1 + '_masked' + NII_EXT)

        if t1_path is not None:
            self.import_t1(t1_path)

        if label_map_path is not None:
            if not self.label_map_path.is_file():
                path.ensure_dir(self.label_map_path)
                self.label_map_path.symlink_to(label_map_path)


    def make_brain_mask(self, force=False):
        if not self.brain_mask_path.is_file() or force:
            seg.get_brain_mask_from_label_map(self.label_map_path,
                                              self.brain_mask_path)


    def mask_t1(self, force=False):
        if not self.t1_masked_path.is_file() or force:
            seg.mask(self.t1_path, self.brain_mask_path, self.t1_masked_path)



class UnsegmentedSubject(Subject):

    def __init__(self, subject_id, t1_age_path=None):
        super().__init__(subject_id)
        self.dir = path.unsegmented_subjects_dir / self.id
        self.transforms_dir = self.dir / 'transforms'
        self.priors_dir = self.dir / 'priors'
        self.segmentation_dir = self.dir / 'segmentation'

        self.t1_path = self.get_t1_path()
        self.age_path = self.dir / (self.id + AGE + TXT_EXT)

        if t1_age_path is not None:
            self.import_t1_id_and_age(t1_age_path)

        ## Registration ##
        self.template_to_t1_affine_path = self.transforms_dir / (
            self.id + '_template_to' + T1 + '_affine' + AFFINE_EXT)
        self.template_to_t1_affine_ff_path = self.transforms_dir / (
            self.id + '_template_to' + T1 + '_affine_ff' + NII_EXT)

        ## Segmentation ##
        self.priors_background_path = self.priors_dir / (
            self.id + PRIORS + '_background' + NII_EXT)
        self.priors_csf_path = self.priors_dir / (
            self.id + PRIORS + '_csf' + NII_EXT)
        self.priors_gm_path = self.priors_dir / (
            self.id + PRIORS + '_gm' + NII_EXT)
        self.priors_wm_path = self.priors_dir / (
            self.id + PRIORS + '_wm' + NII_EXT)
        self.priors_paths_map = {
            const.BACKGROUND: self.priors_background_path,
            const.CSF: self.priors_csf_path,
            const.GREY_MATTER: self.priors_gm_path,
            const.WHITE_MATTER: self.priors_wm_path,
        }

        self.segmentation_background_path = self.segmentation_dir / (
            self.id + SEGMENTATION + '_background' + NII_EXT)
        self.segmentation_csf_path = self.segmentation_dir / (
            self.id + SEGMENTATION + '_csf' + NII_EXT)
        self.segmentation_gm_path = self.segmentation_dir / (
            self.id + SEGMENTATION + '_gm' + NII_EXT)
        self.segmentation_wm_path = self.segmentation_dir / (
            self.id + SEGMENTATION + '_wm' + NII_EXT)
        self.segmentation_paths_map = {
            const.BACKGROUND: self.segmentation_background_path,
            const.CSF: self.segmentation_csf_path,
            const.GREY_MATTER: self.segmentation_gm_path,
            const.WHITE_MATTER: self.segmentation_wm_path,
        }

        self.segmentation_costs_path = self.segmentation_dir / 'costs.npy'


    def __repr__(self):
        return super().__repr__() + f' (unsegmented, age: {self.age})'


    @property
    def age(self):
        return float(self.age_path.read_text())


    def import_t1_id_and_age(self, t1_age_path):
        self.import_t1(t1_age_path)

        path.ensure_dir(self.age_path)
        age = self.get_age_from_image_path(t1_age_path)
        self.age_path.write_text(str(age))


    def get_age_from_image_path(self, t1_age_path):
        pattern = r'.*_(\d+\.?\d?).*'
        age = float(re.match(pattern, t1_age_path.stem).groups()[0])
        return age


    def propagate_priors(self):
        ref_path = self.t1_path
        aff_path = self.template_to_t1_affine_ff_path
        template = get_final_template()
        zipped = zip(template.priors_paths_map.values(),
                     self.priors_paths_map.values())
        for flo_path, res_path in zipped:
            reg.resample(flo_path, ref_path, aff_path, res_path,
                         interpolation=reg.LINEAR)


    def segment(self):
        em = seg.ExpectationMaximisation(self.t1_path,
                                         priors_paths_map=self.priors_paths_map)
        em.run(self.segmentation_paths_map,
               costs_path=self.segmentation_costs_path)
