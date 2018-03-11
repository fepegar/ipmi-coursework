#!/usr/bin/env python3

""" Priors propagation pipeline.

This script performs the priors propagation by registering the template image to
each of the unsegmented images and applying the transformations to the priors
images.
"""

import csv
import multiprocessing as mp

import numpy as np

import ipmi
from ipmi import path
from ipmi import template
from ipmi import registration as reg



class PropagatePriorsPipeline:
    """Pipeline for priors propagation"""
    def __init__(self):
        self.subjects = ipmi.get_segmented_subjects()
        self.subjects.extend(ipmi.get_unsegmented_subjects())


    def register_template_to_subjects(self, free_form=True, force=False):
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

            if (free_form and not cpp_path.is_file()) or force:
                res_path = subject.template_on_t1_affine_ff_path
                # Free-form
                args = ref_path, flo_path
                kwargs = {'trsf_path': cpp_path,
                          'init_trsf_path': aff_path,
                          'res_path': res_path,
                         }
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


    def propagate_priors(self, force=False):
        for subject in self.subjects:
            subject.propagate_priors()


    def segment(self, force=False):
        for subject in self.subjects:
            print('Segmenting', subject)
            subject.segment()


    def dice(self):
        subjects = ipmi.get_segmented_subjects()
        scores_tuples = [subject.dice_scores() for subject in subjects]

        csfs = []
        gms = []
        wms = []
        for scores in scores_tuples:
            csfs.append(scores.csf)
            gms.append(scores.grey_matter)
            wms.append(scores.white_matter)
        scores_array = np.column_stack([csfs, gms, wms])
        scores_array = np.hstack([scores_array,
                                  scores_array.mean(axis=1).reshape(-1, 1)])

        with open(path.dice_report_path, 'w') as csvfile:
            fieldnames = 'Subject', 'CSF', 'Grey matter', 'White matter', 'Mean'
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for subject, (csf, gm, wm, mean) in zip(subjects, scores_array):
                writer.writerow({'Subject': subject.id,
                                 'CSF': csf,
                                 'Grey matter': gm,
                                 'White matter': wm,
                                 'Mean': mean,
                                })
            means = scores_array.mean(axis=0)
            csf, gm, wm, mean = means
            writer.writerow({'Subject': 'Mean',
                             'CSF': csf,
                             'Grey matter': gm,
                             'White matter': wm,
                             'Mean': mean,
                            })



def main():
    force = True
    pipeline = PropagatePriorsPipeline()
    pipeline.register_template_to_subjects(force=force)
    pipeline.propagate_priors(force=force)
    pipeline.segment(force=force)
    pipeline.dice()


if __name__ == '__main__':
    main()
