from pathlib import Path

from numpy import array
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

import ipmi
from ipmi.constants import GREY_MATTER, WHITE_MATTER, BRAIN

sns.set()
sns.set_context('paper')

repo_dir = Path(__file__).parents[2]

subjects = ipmi.get_unsegmented_subjects()
subjects = [s for s in subjects if s.segmentation_em_path.exists()]
print('Subjects:', len(subjects))

ages = array([subject.age for subject in subjects])

print('Computing volumes...')
volumes_tuples = [subject.get_volumes_normalised() for subject in subjects]


print('Saving figure...')
gms = array([t.volumes[GREY_MATTER] for t in volumes_tuples])
wms = array([t.volumes[WHITE_MATTER] for t in volumes_tuples])
brains = array([t.volumes[BRAIN] for t in volumes_tuples])
gms_norm = array([t.normalised_volumes[GREY_MATTER] for t in volumes_tuples])
wms_norm = array([t.normalised_volumes[WHITE_MATTER] for t in volumes_tuples])
brains_norm = array([t.normalised_volumes[BRAIN] for t in volumes_tuples])

def plot(fig, index, x, y, title, norm=False, kind='pearson'):
    ax = fig.add_subplot(2, 3, index)

    if kind == 'pearson':
        cc, p = pearsonr(x, y)
    elif kind == 'spearman':
        cc, p = spearmanr(x, y)
    cc_str = ' (' + r'$\rho$' + f' = {cc:.2f}, '
    if p < 1e-3:
        p_str = r'$p < 10^{-3}$)'
    else:
        p_str = r'$p$' + f' = {p:.3f})'
    title = title + cc_str + p_str
    ax.set_title(title)
    if index > 3:
        ax.set_xlabel('Age (years)')
    if norm:  # y ~ [0, 1]
        y *= 100
        if index == 4:
            ax.set_ylabel('Norm. volume (%)')
    else:  # y in mm^3
        y /= 1000
        if index == 1:
            ax.set_ylabel(r'Volume ($\mathregular{cm^3}$)')
    sns.regplot(x, y, ax=ax, scatter_kws={'s': 5})

figure_path = repo_dir / 'latex' / 'figures' / 'volumes_stats_pearson_ixi.png'

fig = plt.figure()
fig.set_size_inches(8, 4)

plot(fig, 1, ages, gms, 'GM')
plot(fig, 2, ages, wms, 'WM')
plot(fig, 3, ages, brains, 'Brain')
plot(fig, 4, ages, gms_norm, 'GM', norm=True)
plot(fig, 5, ages, wms_norm, 'WM', norm=True)
plot(fig, 6, ages, brains_norm, 'Brain', norm=True)

plt.tight_layout()
fig.savefig(figure_path, dpi=400)



fig = plt.figure()
x, y = ages, wms/gms
title = 'WM/GM'
ax = fig.add_subplot(111)
cc, p = pearsonr(x, y)
title = title + ' (' + r'$\rho$' + f' = {cc:.2f}, ' + r'$p$' + f' = {p:.3f})'
ax.set_title(title)
ax.set_xlabel('Age (years)')
norm = True
index = 1
if norm:  # y ~ [0, 1]
    y *= 100
    ax.set_ylabel('Norm. volume (%)')
else:  # y in mm^3
    y /= 1000
    if index == 1:
        ax.set_ylabel(r'Volume ($\mathregular{cm^3}$)')
sns.regplot(x, y, ax=ax, scatter_kws={'s': 5})


plt.tight_layout()
figure_path = repo_dir / 'latex' / 'figures' / 'volumes_stats_pearson_wm_gm_ixi.png'
fig.savefig(figure_path, dpi=400)
