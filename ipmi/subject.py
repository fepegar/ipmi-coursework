import re
import shutil
from copy import copy
from os.path import relpath
from collections import namedtuple

from . import path
from . import nifti
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
CONFUSION = '_confusion'



class Subject:
    #pylint: disable=E1101

    def __init__(self):

        self.transforms_dir = self.dir / 'transforms'
        self.resampled_dir = self.dir / 'resampled'
        self.priors_dir = self.dir / 'priors'
        self.confusion_dir = self.dir / 'confusion'
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

        self.confusion_background_path = self.confusion_dir / (
            self.id + CONFUSION + '_background' + NII_EXT)
        self.confusion_csf_path = self.confusion_dir / (
            self.id + CONFUSION + '_csf' + NII_EXT)
        self.confusion_gm_path = self.confusion_dir / (
            self.id + CONFUSION + '_gm' + NII_EXT)
        self.confusion_wm_path = self.confusion_dir / (
            self.id + CONFUSION + '_wm' + NII_EXT)
        self.confusion_paths_map = {
            BACKGROUND: self.confusion_background_path,
            CSF: self.confusion_csf_path,
            GREY_MATTER: self.confusion_gm_path,
            WHITE_MATTER: self.confusion_wm_path,
        }

        self.segmentation_em_path = self.segmentation_dir / (
            self.id + SEGMENTATION + '_automatic_label_map' + NII_EXT)

        self.probabilities_path = self.segmentation_dir / (
            self.id + '_probabilities' + NII_EXT)

        self.segmentation_costs_path = self.segmentation_dir / 'costs.npy'
        self.segmentation_costs_plot_path = self.segmentation_dir / 'costs.png'


    def __repr__(self):
        return f'Subject {self.id}'


    def import_t1(self, t1_path):
        if not self.t1_path.is_file():
            path.ensure_dir(self.t1_path)
            self.t1_path.symlink_to(relpath(t1_path, start=self.t1_path.parent))


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


    def segment(self, force=False):
        if self.segmentation_em_path.exists():
            if not force:
                return
            else:
                shutil.rmtree(self.segmentation_dir)
        em = seg.ExpectationMaximisation(self.t1_path,
                                         priors_paths_map=self.priors_paths_map,
                                         write_intermediate=True)
        return em.run(self.segmentation_em_path,
                      costs_path=self.segmentation_costs_path,
                      costs_plot_path=self.segmentation_costs_plot_path,
                      probabilities_path=self.probabilities_path)


    def get_tissues_volumes(self):
        volumes = seg.get_label_map_volumes(self.segmentation_em_path)
        gm = volumes[GREY_MATTER]
        wm = volumes[WHITE_MATTER]
        volumes[BRAIN] = gm + wm
        return volumes


    def get_volumes_normalised(self):
        TissueVolumes = namedtuple('TissueVolumes', ['volumes',
                                                     'normalised_volumes'])
        volumes = self.get_tissues_volumes()
        norm_volumes = copy(volumes)
        icv = volumes[CSF] + volumes[GREY_MATTER] + volumes[WHITE_MATTER]
        norm_volumes[CSF] /= icv
        norm_volumes[GREY_MATTER] /= icv
        norm_volumes[WHITE_MATTER] /= icv
        norm_volumes[BRAIN] /= icv
        result = TissueVolumes(volumes=volumes, normalised_volumes=norm_volumes)
        return result


    def get_uncertainty_image(self):
        probabilities = nifti.load(self.probabilities_path)
        uncertainty = 1 - probabilities.std(axis=3)
        return uncertainty



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
                self.segmentation_manual_path.symlink_to(
                    relpath(segmentation_manual_path,
                            start=self.segmentation_manual_path.parent))


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


    def save_confusion_images(self):
        images = seg.get_confusion_images(self.segmentation_manual_path,
                                          self.segmentation_em_path)
        affine = nifti.load(self.t1_path).affine
        for label, image_path in self.confusion_paths_map.items():
            nifti.save(images[label], affine, image_path,
                       settings={'set_intent': 'vector'})



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
