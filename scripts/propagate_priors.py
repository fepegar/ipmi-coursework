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
        self.subjects = ipmi.get_segmented_subjects()
        self.subjects.extend(ipmi.get_unsegmented_subjects())


    def register_template_to_subjects(self, free_form=True):
        """Register the template to each unsegmented image"""
        flo_path = template.get_final_template().template_image_path
        processes_affine = []
        processes_affine_ff = []
        for subject in self.subjects:
            aff_path = subject.template_to_t1_affine_path
            cpp_path = subject.template_to_t1_affine_ff_path
            ref_path = subject.t1_path

            if not aff_path.is_file():
                # Linear
                args = ref_path, flo_path
                kwargs = {'trsf_path': aff_path}
                process = mp.Process(target=reg.register,
                                     args=args, kwargs=kwargs)
                process.start()
                processes_affine.append(process)

            if free_form and not cpp_path.is_file():
                # Free-form
                args = ref_path, flo_path
                kwargs = {'trsf_path': cpp_path, 'init_trsf_path': aff_path}
                process = mp.Process(target=reg.register_free_form,
                                     args=args, kwargs=kwargs)
                process.start()
                processes_affine_ff.append(process)

        chunks = [processes_affine[i:i + 4]
                  for i in range(len(processes_affine), 4)]
        for chunk in chunks:
            for process in chunk:
                process.join()

        chunks = [processes_affine_ff[i:i + 4]
                  for i in range(len(processes_affine_ff), 4)]
        for chunk in chunks:
            for process in chunk:
                process.join()


    def propagate_priors(self):
        for subject in self.subjects:
            subject.propagate_priors()


def main():
    pipeline = PropagatePriorsPipeline()
    pipeline.register_template_to_subjects()
    pipeline.propagate_priors()

if __name__ == '__main__':
    main()
