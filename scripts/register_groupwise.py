#!/usr/bin/env python3

from ipmi import path
from ipmi import registration as reg
from ipmi.path import get_segmented_images_and_labels

REFERENCE_INDEX = 0

class RegisterGroupwisePipeline:

    def __init__(self):
        self.images_paths, self.labels_paths = get_segmented_images_and_labels()

        self.ref_image_path = self.images_paths[REFERENCE_INDEX]
        self.ref_labels_path = self.labels_paths[REFERENCE_INDEX]

        self.flo_images_paths = self.images_paths[:]
        self.flo_images_paths.remove(self.ref_image_path)

        self.flo_labels_paths = self.labels_paths[:]
        self.flo_labels_paths.remove(self.ref_labels_path)

        self.aff_paths = []
        self.res_images_paths = []
        self.res_labels_paths = []
        ref_name = self.ref_image_path.name.replace('.nii.gz', '')
        for flo_path in self.flo_images_paths:
            flo_name = flo_path.name.replace('.nii.gz', '')

            aff_name = f'{flo_name}_to_{ref_name}.txt'
            aff_path = path.transforms_dir / aff_name
            self.aff_paths.append(aff_path)

            res_image_name = f'{flo_name}_on_{ref_name}_img.nii.gz'
            res_image_path = path.resampled_images_dir / res_image_name
            self.res_images_paths.append(res_image_path)

            res_labels_name = f'{flo_name}_on_{ref_name}_seg.nii.gz'
            res_labels_path = path.resampled_labels_dir / res_labels_name
            self.res_labels_paths.append(res_labels_path)


    def register_all(self):
        for flo_path, aff_path in zip(self.flo_images_paths, self.aff_paths):
            if not aff_path.is_file():
                reg.register(self.ref_image_path, flo_path, trsf_path=aff_path)


    def resample_all(self):
        zipped = zip(self.flo_images_paths,
                     self.flo_labels_paths,
                     self.aff_paths,
                     self.res_images_paths,
                     self.res_labels_paths)
        for flo, labels, aff, res_image, res_labels in zipped:
            if not res_image.is_file():
                reg.resample(flo, self.ref_image_path, aff,
                             res_image, interpolation=reg.SINC)
            if not res_labels.is_file():
                reg.resample(labels, self.ref_image_path, aff,
                             res_labels, interpolation=reg.NN)


    def compute_means(self):
        if not path.template_path.is_file():
            images_paths = [self.ref_image_path] + self.res_images_paths
            reg.compute_mean_image(images_paths, path.template_path)
        if not path.priors_map[0].is_file():
            labels_paths = [self.ref_labels_path] + self.res_labels_paths
            reg.compute_mean_labels(labels_paths, path.priors_map)


def main():
    pipeline = RegisterGroupwisePipeline()
    pipeline.register_all()
    pipeline.resample_all()
    pipeline.compute_means()

if __name__ == '__main__':
    main()
