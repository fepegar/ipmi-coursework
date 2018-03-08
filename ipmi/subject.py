import re

from ipmi import path

NII_EXT = '.nii.gz'
TXT_EXT = '.txt'

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



class SegmentedSubject(Subject):

    def __init__(self, subject_id, t1_path=None, label_map_path=None):
        super().__init__(subject_id)
        self.dir = path.segmented_subjects_dir / self.id
        self.t1_path = self.get_t1_path()
        self.label_map_path = self.dir / (self.id + LABEL_MAP + NII_EXT)

        if t1_path is not None:
            self.import_t1(t1_path)

        if label_map_path is not None:
            if not self.label_map_path.is_file():
                path.ensure_dir(self.label_map_path)
                self.label_map_path.symlink_to(label_map_path)



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