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
        # The zip object must be converted to list because
        # we iterate twice over it
        zipped_refs_affines = list(zip(self.unsegmented_paths, self.aff_paths))
        zipped_priors_paths = zip(path.priors_map.values(), path.priors_dirs)
        for flo_path, res_prior_dir in zipped_priors_paths:
            flo_name = flo_path.name.replace('.nii.gz', '')
            for ref_path, aff_path in zipped_refs_affines:
                ref_name = '_'.join(ref_path.name.split('_')[:2])
                res_path = res_prior_dir / f'{flo_name}_on_{ref_name}.nii.gz'
                if not res_path.is_file():
                    reg.resample(flo_path, ref_path, aff_path,
                                 res_path, interpolation=reg.LINEAR)



def main():
    pipeline = PropagatePriorsPipeline()
    pipeline.register_template_to_unsegmented()
    pipeline.propagate_priors()

if __name__ == '__main__':
    main()
