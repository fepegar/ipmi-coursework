#!/usr/bin/env python3

""" Priors propagation pipeline.

This script performs the priors propagation by registering the template image to
each of the unsegmented images and applying the transformations to the priors
images.
"""

import multiprocessing as mp

import ipmi
from ipmi import path
from ipmi import template
from ipmi import registration as reg



class PropagatePriorsPipeline:
    """Pipeline for priors propagation"""
    def __init__(self):
        self.unsegmented_subjects = ipmi.get_unsegmented_subjects()


    def register_template_to_unsegmented(self):
        """Register the template to each unsegmented image"""
        flo_path = template.get_final_template().template_image_path
        processes = []
        for subject in self.unsegmented_subjects:
            aff_path = subject.template_to_t1_path
            ref_path = subject.t1_path

            if not aff_path.is_file():
                args = ref_path, flo_path
                kwargs = {'trsf_path': aff_path}
                process = mp.Process(target=reg.register,
                                     args=args, kwargs=kwargs)
                process.start()
                processes.append(process)

        for process in processes:
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
