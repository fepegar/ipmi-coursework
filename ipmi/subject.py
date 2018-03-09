import re

from ipmi import path
from . import segmentation as seg

NII_EXT = '.nii.gz'
TXT_EXT = '.txt'
AFFINE_EXT = TXT_EXT

T1 = '_t1'
LABEL_MAP = '_label_map'
AGE = '_age'



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
        self.t1_path = self.get_t1_path()
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
