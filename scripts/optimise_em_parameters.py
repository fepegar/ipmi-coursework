from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import ipmi
from ipmi.path import output_dir
from ipmi import constants as const
from ipmi import segmentation as seg

max_polynomial_order = 3

# betas = np.array([0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 1, 2])
betas = np.logspace(-2, np.log10(2), 8)
orders = np.arange(max_polynomial_order + 1)

subject = ipmi.get_segmented_subjects()[0]
segmentations_dir = output_dir / 'seg_optimisation'
dices_path = segmentations_dir / 'dices.npy'
figure_path = segmentations_dir / 'dices.png'
paths_dict = OrderedDict()

# RUN SEGMENTATIONS
for beta in betas:
    for order in orders:
        name = f'{subject.id}_order_{order}_beta_{beta:.3f}_seg'
        name = name.replace('.', '_')
        filename = name + '.nii.gz'
        segmentation_path = segmentations_dir / filename
        if order not in paths_dict:
            paths_dict[order] = OrderedDict()
        paths_dict[order][beta] = segmentation_path
        if segmentation_path.is_file(): continue
        em = seg.ExpectationMaximisation(subject.t1_path,
                                         priors_paths_map=subject.priors_paths_map,
                                         write_intermediate=False)
        em.set_beta(beta)
        em.set_inu_polynomial_order(order)
        print(f'SEGMENTING WITH N = {order} and BETA = {beta}')
        em.run(segmentation_path)

## DICES
ground_truth_path = subject.segmentation_manual_path

if dices_path.is_file():
    dices = np.load(str(dices_path))
else:
    dices = np.zeros((max_polynomial_order + 1, len(betas), 3))
    for ind_beta, beta in enumerate(betas):
        for order in orders:
            print(f'DICE for N = {order} and BETA = {beta}')
            auto_seg_path = paths_dict[order][beta]
            scores = seg.label_map_dice_scores(ground_truth_path, auto_seg_path)
            csf = scores[const.CSF]
            gm = scores[const.GREY_MATTER]
            wm = scores[const.WHITE_MATTER]
            dices[order, ind_beta] = csf, gm, wm
    np.save(str(dices_path), dices)

dice_mean = dices.mean(axis=2)

def plot_dices(fig, axis, dices_matrix, title):
    im = axis.imshow(dices_matrix)
    fig.colorbar(im, ax=axis)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_ylabel('Polynomial order')
    axis.set_xlabel('beta')
    # axis.set_xticks(betas, list(map(str, betas)))


fig = plt.figure()
# plot_dices(fig, fig.add_subplot(221), dices[..., 0], 'CSF')
# plot_dices(fig, fig.add_subplot(222), dices[..., 1], 'GM')
# plot_dices(fig, fig.add_subplot(223), dices[..., 2], 'WM')
# plot_dices(fig, fig.add_subplot(224), dice_mean, 'Mean')
# plt.tight_layout()

from scipy import interpolate
x = betas
y = orders
z = dice_mean
xx, yy = np.meshgrid(x, y)
f = interpolate.interp2d(x, y, z, kind='linear')
xnew = np.logspace(-2, 1)
ynew = np.linspace(0, orders.max())
znew = f(xnew, ynew)
axis = fig.add_subplot(111)
axis.set_xscale('log')
# axis.imshow(znew, extent=(0, betas.max(), 0, orders.max()), origin='lower')
axis.pcolor(xnew, ynew, znew)
i_max, j_max = np.where(znew == znew.max())
x_max = xnew[j_max]
y_max = ynew[i_max]
axis.scatter(x_max, y_max)
axis.scatter(xx, yy, s=5)
# fig.savefig(figure_path, dpi=200)
plt.show()
