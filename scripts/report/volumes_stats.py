from pathlib import Path

from numpy import array
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

import ipmi
from ipmi.constants import GREY_MATTER, WHITE_MATTER, BRAIN

sns.set()
sns.set_context('paper')

def plot(fig, index, x, y, title, norm=False):
    ax = fig.add_subplot(2, 3, index)
    cc, p = pearsonr(x, y)
    title = title + ' (' + r'$R$' + f' = {cc:.2f}, ' + r'$p$' + f' = {p:.3f})'
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
    sns.regplot(x, y, ax=ax)


repo_dir = Path(__file__).parents[2]
figure_path = repo_dir / 'latex' / 'figures' / 'volumes_stats.png'


subjects = ipmi.get_unsegmented_subjects()
subjects = [s for s in subjects if s.segmentation_em_path.exists()]

ages = array([subject.age for subject in subjects])

print('Computing volumes...')
volumes_tuples = [subject.get_volumes_normalised() for subject in subjects]

gms = array([t.volumes[GREY_MATTER] for t in volumes_tuples])
wms = array([t.volumes[WHITE_MATTER] for t in volumes_tuples])
brains = array([t.volumes[BRAIN] for t in volumes_tuples])
gms_norm = array([t.normalised_volumes[GREY_MATTER] for t in volumes_tuples])
wms_norm = array([t.normalised_volumes[WHITE_MATTER] for t in volumes_tuples])
brains_norm = array([t.normalised_volumes[BRAIN] for t in volumes_tuples])

print('Saving figure...')
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
