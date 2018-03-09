from . import path
from . import constants as const
from . import registration as reg

NII_EXT = '.nii.gz'
PRIORS = '_priors'

class Template:

    def __init__(self, template_id):
        self.id = template_id
        self.dir = path.templates_dir / self.id
        self.template_image_path = self.dir / (
            self.id + '_template_image' + NII_EXT)
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


    def exists(self):
        return self.template_image_path.is_file()


    def generate(self, images_paths, labels_paths):
        reg.compute_mean_image(images_paths, self.template_image_path)
        reg.compute_mean_labels(labels_paths, self.priors_paths_map)
