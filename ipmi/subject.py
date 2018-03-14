import re
from copy import copy
from collections import namedtuple

from . import path
from . import segmentation as seg
from . import registration as reg
from .template import get_final_template
from .constants import BACKGROUND, CSF, GREY_MATTER, WHITE_MATTER, BRAIN

NII_EXT = '.nii.gz'
TXT_EXT = '.txt'
AFFINE_EXT = TXT_EXT

T1 = '_t1'
LABEL_MAP = '_label_map'
AGE = '_age'
SEGMENTATION = '_segmentation'
MANUAL = '_manual'
PRIORS = '_priors'


class Subject:
    #pylint: disable=E1101

    def __init__(self):

        self.transforms_dir = self.dir / 'transforms'
        self.resampled_dir = self.dir / 'resampled'
        self.priors_dir = self.dir / 'priors'
        self.segmentation_dir = self.dir / 'segmentation'

        self.t1_path = self.dir / (self.id + T1 + NII_EXT)

        ## Registration ##
        self.template_to_t1_affine_path = self.transforms_dir / (
            self.id + '_template_to' + T1 + '_affine' + AFFINE_EXT)
        self.template_to_t1_affine_ff_path = self.transforms_dir / (
            self.id + '_template_to' + T1 + '_affine_ff' + NII_EXT)
        self.template_on_t1_affine_ff_path = self.resampled_dir / (
            self.id + '_template_on' + T1 + '_affine_ff' + NII_EXT)

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
            BACKGROUND: self.priors_background_path,
            CSF: self.priors_csf_path,
            GREY_MATTER: self.priors_gm_path,
            WHITE_MATTER: self.priors_wm_path,
        }

        self.segmentation_em_path = self.segmentation_dir / (
            self.id + SEGMENTATION + '_automatic' + NII_EXT)

        self.segmentation_costs_path = self.segmentation_dir / 'costs.npy'
        self.segmentation_costs_plot_path = self.segmentation_dir / 'costs.png'


    def __repr__(self):
        return f'Subject {self.id}'


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
            self.id + SEGMENTATION + MANUAL + f'_on_{template.id}' + NII_EXT)
        return res_path


    def propagate_priors(self, non_linear=True, force=False):
        ref_path = self.t1_path
        if non_linear:
            aff_path = self.template_to_t1_affine_ff_path
        else:
            aff_path = self.template_to_t1_affine_path
        template = get_final_template()
        zipped = zip(template.priors_paths_map.values(),
                     self.priors_paths_map.values())
        for flo_path, res_path in zipped:
            if not res_path.is_file() or force:
                reg.resample(flo_path, ref_path, aff_path, res_path,
                             interpolation=reg.LINEAR)


    def segment(self):
        em = seg.ExpectationMaximisation(self.t1_path,
                                         priors_paths_map=self.priors_paths_map,
                                         write_intermediate=True)
        return em.run(self.segmentation_em_path,
                      costs_path=self.segmentation_costs_path,
                      costs_plot_path=self.segmentation_costs_plot_path)


    def get_tissues_volumes(self):
        volumes = seg.get_label_map_volumes(self.segmentation_em_path)
        csf = volumes[CSF]
        gm = volumes[GREY_MATTER]
        wm = volumes[WHITE_MATTER]
        volumes[BRAIN] = csf + gm + wm
        return volumes


    def get_volumes_normalised(self):
        TissueVolumes = namedtuple('TissueVolumes', ['volumes',
                                                     'normalised_volumes'])
        volumes = self.get_tissues_volumes()
        norm_volumes = copy(volumes)
        del norm_volumes[BRAIN]
        norm_volumes[CSF] /= volumes[BRAIN]
        norm_volumes[GREY_MATTER] /= volumes[BRAIN]
        norm_volumes[WHITE_MATTER] /= volumes[BRAIN]
        result = TissueVolumes(volumes=volumes, normalised_volumes=norm_volumes)
        return result



class SegmentedSubject(Subject):

    def __init__(self, subject_id, t1_path=None, segmentation_manual_path=None):
        self.id = subject_id
        self.dir = path.segmented_subjects_dir / self.id
        super().__init__()

        self.segmentation_manual_path = self.dir / (
            self.id + SEGMENTATION + MANUAL + NII_EXT)
        self.brain_mask_path = self.dir / (self.id + '_brain_mask' + NII_EXT)
        self.t1_masked_path = self.dir / (self.id + T1 + '_masked' + NII_EXT)

        if t1_path is not None:
            self.import_t1(t1_path)

        if segmentation_manual_path is not None:
            if not self.segmentation_manual_path.is_file():
                path.ensure_dir(self.segmentation_manual_path)
                self.segmentation_manual_path.symlink_to(segmentation_manual_path)


    def __repr__(self):
        return super().__repr__() + ' (segmented)'


    def make_brain_mask(self, force=False):
        if not self.brain_mask_path.is_file() or force:
            seg.get_brain_mask_from_label_map(self.segmentation_manual_path,
                                              self.brain_mask_path)


    def mask_t1(self, force=False):
        if not self.t1_masked_path.is_file() or force:
            seg.mask(self.t1_path, self.brain_mask_path, self.t1_masked_path)


    def dice_scores(self):
        DiceScores = namedtuple('DiceScores',
                                ['csf', 'grey_matter', 'white_matter'])
        scores_dict = seg.label_map_dice_scores(self.segmentation_manual_path,
                                                self.segmentation_em_path)
        scores_tuple = DiceScores(csf=scores_dict[CSF],
                                  grey_matter=scores_dict[GREY_MATTER],
                                  white_matter=scores_dict[WHITE_MATTER])
        return scores_tuple



class UnsegmentedSubject(Subject):

    def __init__(self, subject_id, t1_age_path=None):
        self.id = subject_id
        self.dir = path.unsegmented_subjects_dir / self.id
        super().__init__()
        self.age_path = self.dir / (self.id + AGE + TXT_EXT)

        if t1_age_path is not None:
            self.import_t1_id_and_age(t1_age_path)


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
