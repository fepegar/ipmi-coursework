#!/usr/bin/env python3

import multiprocessing as mp

import ipmi
from ipmi import path
from ipmi import registration as reg
from ipmi import Template, SegmentedSubject, UnsegmentedSubject



class RegisterGroupwiseIterativePipeline:

    def __init__(self):
        self.import_all_subjects()
        self.segmented_subjects = ipmi.get_segmented_subjects()
        self.rigid_template = Template('rigid')


    def import_all_subjects(self):
        segmented_paths, labels_paths = path.get_segmented_images_and_labels()
        for img_path, labels_path in zip(segmented_paths, labels_paths):
            subject_id = img_path.name.split('_')[1]
            SegmentedSubject(subject_id,
                             t1_path=img_path, label_map_path=labels_path)

        unsegmented_paths, _ = path.get_unsegmented_images_and_ages()
        for img_path in unsegmented_paths:
            subject_id = img_path.name.split('_')[1]
            UnsegmentedSubject(subject_id, t1_age_path=img_path)


    def create_template_rigid(self, reference_id):
        print('Creating rigid template')
        template = self.rigid_template
        reference_subject = ipmi.get_subject_by_id(self.segmented_subjects,
                                                   reference_id)
        reference_subject.make_brain_mask()
        ref_path = reference_subject.t1_path
        ref_mask_path = reference_subject.brain_mask_path

        ref_res_img_path = reference_subject.get_image_on_template_path(
            template)
        path.ensure_dir(ref_res_img_path)
        if not ref_res_img_path.is_file():
            ref_res_img_path.symlink_to(ref_path)

        ref_res_labels_path = reference_subject.get_label_map_on_template_path(
            template)
        path.ensure_dir(ref_res_labels_path)
        if not ref_res_labels_path.is_file():
            ref_res_labels_path.symlink_to(reference_subject.label_map_path)

        # Run registrations in parallel
        processes = []
        for subject in self.segmented_subjects:
            if subject is reference_subject:
                continue
            subject.make_brain_mask()
            flo_path = subject.t1_path
            flo_mask_path = subject.brain_mask_path
            aff_path = subject.get_affine_to_template_path(template)
            res_t1_path = subject.get_image_on_template_path(template)
            res_label_map_path = subject.get_label_map_on_template_path(
                template)
            label_map_path = subject.label_map_path

            if aff_path.is_file():
                continue

            args = ref_path, flo_path
            kwargs = {
                'trsf_path': aff_path,
                'ref_mask_path': ref_mask_path,
                'flo_mask_path': flo_mask_path,
            }
            process = mp.Process(target=reg.register, args=args, kwargs=kwargs)
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        # Resample images and label maps
        for subject in self.segmented_subjects:
            subject.mask_t1()
            if subject is reference_subject:
                continue
            flo_path = subject.t1_masked_path
            aff_path = subject.get_affine_to_template_path(template)
            res_t1_path = subject.get_image_on_template_path(template)
            res_label_map_path = subject.get_label_map_on_template_path(
                template)
            label_map_path = subject.label_map_path

            if res_t1_path.is_file():
                continue

            reg.resample(flo_path, ref_path, aff_path,
                         res_t1_path, interpolation=reg.SINC)
            reg.resample(label_map_path, ref_path, aff_path,
                         res_label_map_path, interpolation=reg.NEAREST)

        # Create template and priors from resampled images
        if not self.rigid_template.exists():
            images_paths = [subject.get_image_on_template_path(template)
                            for subject
                            in self.segmented_subjects]

            labels_paths = [subject.get_label_map_on_template_path(template)
                            for subject
                            in self.segmented_subjects]

            self.rigid_template.generate(images_paths, labels_paths)


    def create_templates_affine(self, iterations):
        reference_template = self.rigid_template
        for i in range(iterations):
            print(f'Running iteration {i} for affine template')
            affine_template = Template(f'affine_template_iter_{i}')
            ref_path = reference_template.template_image_path

            # Register images in parallel
            processes = []
            for subject in self.segmented_subjects:
                flo_path = subject.t1_path
                flo_mask_path = subject.brain_mask_path
                aff_path = subject.get_affine_to_template_path(affine_template)
                init_trsf_path = subject.get_affine_to_template_path(
                    reference_template)
                if aff_path.is_file():
                    continue

                args = ref_path, flo_path
                kwargs = {
                    'trsf_path': aff_path,
                    'flo_mask_path': flo_mask_path,
                }
                if init_trsf_path.is_file():
                    kwargs['init_trsf_path'] = init_trsf_path

                process = mp.Process(target=reg.register, args=args, kwargs=kwargs)
                process.start()
                processes.append(process)

            for process in processes:
                process.join()


            # Resample images and labels
            for subject in self.segmented_subjects:
                flo_path = subject.t1_masked_path
                aff_path = subject.get_affine_to_template_path(affine_template)

                res_t1_path = subject.get_image_on_template_path(
                    affine_template)
                res_label_map_path = subject.get_label_map_on_template_path(
                    affine_template)
                label_map_path = subject.label_map_path

                if res_t1_path.is_file():
                    continue

                reg.resample(flo_path, ref_path, aff_path,
                             res_t1_path, interpolation=reg.SINC)
                reg.resample(label_map_path, ref_path, aff_path,
                             res_label_map_path, interpolation=reg.NEAREST)

            # Make template and priors from resampled images
            if not affine_template.exists():
                images_paths = [
                    subject.get_image_on_template_path(affine_template)
                    for subject
                    in self.segmented_subjects
                ]

                labels_paths = [
                    subject.get_label_map_on_template_path(affine_template)
                    for subject
                    in self.segmented_subjects
                ]

                affine_template.generate(images_paths, labels_paths)

            reference_template = affine_template


def main():
    pipeline = RegisterGroupwiseIterativePipeline()
    pipeline.create_template_rigid(reference_id='0')
    pipeline.create_templates_affine(iterations=10)


if __name__ == '__main__':
    main()