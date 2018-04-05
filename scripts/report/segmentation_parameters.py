import csv
import pathlib

import pandas
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

import ipmi
from ipmi import segmentation as seg
subject = ipmi.get_segmented_subjects()[0]

dices = []

segmentation_dir = subject.dir / 'seg'
em = seg.ExpectationMaximisation(subject.t1_path,
                                 write_intermediate=True,
                                 use_mrf=False,
                                 use_bias_correction=False,
                                 num_classes=4)

subject.segmentation_em_path = segmentation_dir / subject.segmentation_em_path.name
subject.probabilities_path = segmentation_dir / subject.probabilities_path.name
subject.uncertainty_img_path = segmentation_dir / subject.uncertainty_img_path.name
subject.segmentation_collage_path = segmentation_dir / subject.segmentation_collage_path.name
subject.confusion_background_path = segmentation_dir / subject.confusion_background_path.name
subject.confusion_csf_path = segmentation_dir / subject.confusion_csf_path.name
subject.confusion_gm_path = segmentation_dir / subject.confusion_gm_path.name
subject.confusion_wm_path = segmentation_dir / subject.confusion_wm_path.name
subject.confusion_paths_map[0] = subject.confusion_background_path
subject.confusion_paths_map[1] = subject.confusion_csf_path
subject.confusion_paths_map[2] = subject.confusion_gm_path
subject.confusion_paths_map[3] = subject.confusion_wm_path
# em.run(subject.segmentation_em_path, probabilities_path=subject.probabilities_path)
# subject.save_confusion_images()
# subject.make_uncertainty_image()
# subject.make_segmentation_collage()
print(segmentation_dir)
dices.append(subject.dice_scores())
print()


segmentation_dir = subject.dir / 'seg_priors'
em = seg.ExpectationMaximisation(subject.t1_path,
                                 priors_paths_map=subject.priors_paths_map,
                                 write_intermediate=True,
                                 use_mrf=False,
                                 use_bias_correction=False)

subject.segmentation_em_path = segmentation_dir / subject.segmentation_em_path.name
subject.probabilities_path = segmentation_dir / subject.probabilities_path.name
subject.uncertainty_img_path = segmentation_dir / subject.uncertainty_img_path.name
subject.segmentation_collage_path = segmentation_dir / subject.segmentation_collage_path.name
subject.confusion_background_path = segmentation_dir / subject.confusion_background_path.name
subject.confusion_csf_path = segmentation_dir / subject.confusion_csf_path.name
subject.confusion_gm_path = segmentation_dir / subject.confusion_gm_path.name
subject.confusion_wm_path = segmentation_dir / subject.confusion_wm_path.name
subject.confusion_paths_map[0] = subject.confusion_background_path
subject.confusion_paths_map[1] = subject.confusion_csf_path
subject.confusion_paths_map[2] = subject.confusion_gm_path
subject.confusion_paths_map[3] = subject.confusion_wm_path
# em.run(subject.segmentation_em_path, probabilities_path=subject.probabilities_path)
# subject.save_confusion_images()
# subject.make_uncertainty_image()
# subject.make_segmentation_collage()
print(segmentation_dir)
dices.append(subject.dice_scores())
print()



segmentation_dir = subject.dir / 'seg_priors_mrf_beta_small'
em = seg.ExpectationMaximisation(subject.t1_path,
                                 priors_paths_map=subject.priors_paths_map,
                                 write_intermediate=True,
                                 use_mrf=True,
                                 use_bias_correction=False)

subject.segmentation_em_path = segmentation_dir / subject.segmentation_em_path.name
subject.probabilities_path = segmentation_dir / subject.probabilities_path.name
subject.uncertainty_img_path = segmentation_dir / subject.uncertainty_img_path.name
subject.segmentation_collage_path = segmentation_dir / subject.segmentation_collage_path.name
subject.confusion_background_path = segmentation_dir / subject.confusion_background_path.name
subject.confusion_csf_path = segmentation_dir / subject.confusion_csf_path.name
subject.confusion_gm_path = segmentation_dir / subject.confusion_gm_path.name
subject.confusion_wm_path = segmentation_dir / subject.confusion_wm_path.name
subject.confusion_paths_map[0] = subject.confusion_background_path
subject.confusion_paths_map[1] = subject.confusion_csf_path
subject.confusion_paths_map[2] = subject.confusion_gm_path
subject.confusion_paths_map[3] = subject.confusion_wm_path
# em.run(subject.segmentation_em_path, probabilities_path=subject.probabilities_path)
# subject.save_confusion_images()
# subject.make_uncertainty_image()
# subject.make_segmentation_collage()
print(segmentation_dir)
dices.append(subject.dice_scores())
print()



segmentation_dir = subject.dir / 'seg_priors_mrf_beta_small_bias_correction'
em = seg.ExpectationMaximisation(subject.t1_path,
                                 priors_paths_map=subject.priors_paths_map,
                                 write_intermediate=True,
                                 use_mrf=True,
                                 use_bias_correction=True)
subject.segmentation_em_path = segmentation_dir / subject.segmentation_em_path.name
subject.probabilities_path = segmentation_dir / subject.probabilities_path.name
subject.uncertainty_img_path = segmentation_dir / subject.uncertainty_img_path.name
subject.segmentation_collage_path = segmentation_dir / subject.segmentation_collage_path.name
subject.confusion_background_path = segmentation_dir / subject.confusion_background_path.name
subject.confusion_csf_path = segmentation_dir / subject.confusion_csf_path.name
subject.confusion_gm_path = segmentation_dir / subject.confusion_gm_path.name
subject.confusion_wm_path = segmentation_dir / subject.confusion_wm_path.name
subject.confusion_paths_map[0] = subject.confusion_background_path
subject.confusion_paths_map[1] = subject.confusion_csf_path
subject.confusion_paths_map[2] = subject.confusion_gm_path
subject.confusion_paths_map[3] = subject.confusion_wm_path
# em.run(subject.segmentation_em_path, probabilities_path=subject.probabilities_path)
# subject.save_confusion_images()
# subject.make_uncertainty_image()
# subject.make_segmentation_collage()
print(segmentation_dir)
dices.append(subject.dice_scores())
print()



segmentation_dir = subject.dir / 'seg_priors_mrf_beta_large'
em = seg.ExpectationMaximisation(subject.t1_path,
                                 priors_paths_map=subject.priors_paths_map,
                                 write_intermediate=True,
                                 use_mrf=True,
                                 use_bias_correction=False)
em.set_beta(2)
subject.segmentation_em_path = segmentation_dir / subject.segmentation_em_path.name
subject.probabilities_path = segmentation_dir / subject.probabilities_path.name
subject.uncertainty_img_path = segmentation_dir / subject.uncertainty_img_path.name
subject.segmentation_collage_path = segmentation_dir / subject.segmentation_collage_path.name
subject.confusion_background_path = segmentation_dir / subject.confusion_background_path.name
subject.confusion_csf_path = segmentation_dir / subject.confusion_csf_path.name
subject.confusion_gm_path = segmentation_dir / subject.confusion_gm_path.name
subject.confusion_wm_path = segmentation_dir / subject.confusion_wm_path.name
subject.confusion_paths_map[0] = subject.confusion_background_path
subject.confusion_paths_map[1] = subject.confusion_csf_path
subject.confusion_paths_map[2] = subject.confusion_gm_path
subject.confusion_paths_map[3] = subject.confusion_wm_path
# em.run(subject.segmentation_em_path, probabilities_path=subject.probabilities_path)
# subject.save_confusion_images()
# subject.make_uncertainty_image()
# subject.make_segmentation_collage()
print(segmentation_dir)
dices.append(subject.dice_scores())
print()



##########

tmp_csv = '/tmp/dices.csv'
experiments = list('ABCDE')
with open(tmp_csv, 'w') as csvfile:
    fieldnames = 'Experiment', 'Dice score', 'Tissue'
    tissues = 'CSF', 'Grey matter', 'White matter'
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for experiment, dice in zip(experiments, dices):
        for i, tissue in enumerate(tissues):
            writer.writerow({'Experiment': experiment,
                             'Dice score': dice[i],
                             'Tissue': tissue,
                            })
        mean = np.array(dice).mean()
        writer.writerow({'Experiment': experiment,
                         'Dice score': mean,
                         'Tissue': 'Mean',
                        })





sns.set()
sns.set_context('paper')

repo_dir = pathlib.Path(__file__).parents[2]
figure_path = repo_dir / 'latex' / 'figures' / 'experiments_dices_bars.png'

grid = sns.factorplot(x='Experiment', y='Dice score',
                      data=pandas.read_csv(tmp_csv), hue='Tissue',
                      kind='bar')

grid.ax.set_ybound(lower=0.35)

fig = plt.gcf()
fig.set_size_inches(8, 4)
fig.savefig(figure_path, dpi=400)
