#!/usr/bin/env python3

import csv

from scipy.stats import pearsonr

import ipmi
from ipmi.constants import GREY_MATTER, WHITE_MATTER, BRAIN

def compute_and_write_row(writer, x, y, name):
    cc, p = pearsonr(x, y)
    writer.writerow(
        {'Measurement': name, 'Pearson CC': f'{cc:.2f}', 'p': f'{p:.3f}'})


if __name__ == '__main__':
    subjects = ipmi.get_unsegmented_subjects()
    ages = [subject.age for subject in subjects]
    volumes_tuples = [subject.get_volumes_normalised() for subject in subjects]

    icvs = [tup.volumes[BRAIN] for tup in volumes_tuples]
    gms = [tup.volumes[GREY_MATTER] for tup in volumes_tuples]
    wms = [tup.volumes[WHITE_MATTER] for tup in volumes_tuples]
    gms_norm = [tup.normalised_volumes[GREY_MATTER] for tup in volumes_tuples]
    wms_norm = [tup.normalised_volumes[WHITE_MATTER] for tup in volumes_tuples]

    csv_path = ipmi.path.correlations_report_path

    with open(str(csv_path), 'w') as csvfile:
        fieldnames = 'Measurement', 'Pearson CC', 'p'
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        compute_and_write_row(writer, icvs, ages, 'ICV')
        compute_and_write_row(writer, gms, ages, 'GM')
        compute_and_write_row(writer, wms, ages, 'WM')
        compute_and_write_row(writer, gms_norm, ages, 'GM norm')
        compute_and_write_row(writer, wms_norm, ages, 'WM norm')
