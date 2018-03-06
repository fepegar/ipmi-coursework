#!/usr/bin/env python3

""" Priors propagation pipeline.

This script performs the priors propagation by registering the template image to
each of the unsegmented images and applying the transformations to the priors
images.
"""

from ipmi import path
from ipmi import registration as reg

class PropagatePriorsPipeline:
    """Pipeline for priors propagation"""
    def __init__(self):
        self.unsegmented_paths, _ = path.get_unsegmented_images_and_ages()
        self.aff_paths = []
        for image_path in self.unsegmented_paths:
            image_name = '_'.join(image_path.name.split('_')[:2])
            aff_name = f'template_to_{image_name}.txt'
            aff_path = path.transforms_from_template_dir / aff_name
            self.aff_paths.append(aff_path)


    def register_template_to_unsegmented(self):
        """Register the template to each unsegmented image"""
        flo_path = path.template_path
        zipped = zip(self.unsegmented_paths, self.aff_paths)
        for unsegmented_path, aff_path in zipped:
            if not aff_path.is_file():
                reg.register(unsegmented_path,
                             flo_path,
                             trsf_path=aff_path)


    def propagate_priors(self):
        return



def main():
    pipeline = PropagatePriorsPipeline()
    pipeline.register_template_to_unsegmented()
    pipeline.propagate_priors()

if __name__ == '__main__':
    main()
