#!/usr/bin/env python3

""" Priors propagation pipeline.

This script performs the priors propagation by registering the template image to
each of the unsegmented images and applying the transformations to the priors
images.
"""

import multiprocessing as mp

import ipmi
from ipmi import template
from ipmi import registration as reg



class PropagatePriorsPipeline:
    """Pipeline for priors propagation"""
    def __init__(self):
        self.unsegmented_subjects = ipmi.get_unsegmented_subjects()


    def register_template_to_unsegmented(self, free_form=True):
        """Register the template to each unsegmented image"""
        flo_path = template.get_final_template().template_image_path
        processes_affine = []
        processes_affine_ff = []
        for subject in self.unsegmented_subjects:
            aff_path = subject.template_to_t1_affine_path
            cpp_path = subject.template_to_t1_affine_ff_path
            ref_path = subject.t1_path

            if aff_path.is_file():
                continue

            # Linear
            args = ref_path, flo_path
            kwargs = {'trsf_path': aff_path}
            process = mp.Process(target=reg.register,
                                 args=args, kwargs=kwargs)
            process.start()
            processes_affine.append(process)

            if not free_form:
                continue
            # Free-form
            args = ref_path, flo_path
            kwargs = {'trsf_path': cpp_path, 'init_trsf_path': aff_path}
            process = mp.Process(target=reg.register_free_form,
                                 args=args, kwargs=kwargs)
            process.start()
            processes_affine_ff.append(process)

        for process in processes_affine:
            process.join()

        for process in processes_affine_ff:
            process.join()


    def propagate_priors(self):
        for subject in self.unsegmented_subjects:
            subject.propagate_priors()


def main():
    pipeline = PropagatePriorsPipeline()
    pipeline.register_template_to_unsegmented()
    pipeline.propagate_priors()

if __name__ == '__main__':
    main()
