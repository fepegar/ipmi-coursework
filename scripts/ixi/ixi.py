from pathlib import Path

import pandas
import numpy as np

import ipmi
from ipmi.constants import ROOT_DIR
from ipmi import path, UnsegmentedSubject

ixi_images_dir = ROOT_DIR / 'IXI-T1'
ixi_demographic_data = pandas.read_csv(ROOT_DIR / 'IXI.csv')

data = [(ixi_id, age)
        for (ixi_id, age)
        in zip(ixi_demographic_data.IXI_ID, ixi_demographic_data.AGE)
        if not np.isnan(age)
       ]

data = dict(data)
ixi_images_paths = sorted(ixi_images_dir.glob('*.nii.gz'))
filtered_paths = []
for ixi_path in ixi_images_paths:
    path_id = int(ixi_path.stem.split('-')[0][3:])
    if path_id in data:
        path_id_str = f'{path_id:03d}'
        fn = f'img_{path_id_str}_age_{data[path_id]:.1f}.nii.gz'
        print(fn)
        fp = path.unsegmented_images_dir / fn
        path.ensure_dir(fp)
        filtered_paths.append(fp)
        if not fp.exists():
            fp.symlink_to(ixi_path)
        subject = UnsegmentedSubject(path_id_str, t1_age_path=fp)
