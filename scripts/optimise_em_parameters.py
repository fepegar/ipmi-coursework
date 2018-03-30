from collections import OrderedDict

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns

import ipmi
from ipmi.path import output_dir
from ipmi import constants as const
from ipmi import segmentation as seg

sns.set_context("paper")
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
        if segmentation_path.is_file():
            continue
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
            auto_seg_path = paths_dict[order][beta]
            scores = seg.label_map_dice_scores(ground_truth_path, auto_seg_path)
            csf = scores[const.CSF]
            gm = scores[const.GREY_MATTER]
            wm = scores[const.WHITE_MATTER]
            dices[order, ind_beta] = csf, gm, wm
            # print(f'DICE for N = {order} and BETA = {beta:.2f}: {np.mean((csf, gm, wm)):.3f}')
    np.save(str(dices_path), dices)

dice_mean = dices.mean(axis=2)


def plot_dices(fig, axis, dices_matrix, title):
    x = betas
    y = orders
    z = dices_matrix
    xx, yy = np.meshgrid(x, y)
    f = interpolate.interp2d(x, y, z, kind='cubic')
    xnew = np.logspace(-2, np.log10(2), num=50)
    ynew = np.linspace(0, orders.max(), num=50)
    znew = f(xnew, ynew)

    axis.set_xscale('log')
    # axis.grid()
    # im = axis.pcolor(xnew, ynew, znew)
    im = axis.contour(xnew, ynew, znew, cmap='viridis')#, vmin=dices.min(), vmax=dices.max())
    fig.colorbar(im, ax=axis)

    i_max, j_max = np.where(z == z.max())
    x_max = x[j_max]
    y_max = y[i_max]
    axis.scatter(x_max, y_max, label='Max sampled Dice')

    i_max, j_max = np.where(znew == znew.max())
    x_max = xnew[j_max]
    y_max = ynew[i_max]
    axis.scatter(x_max, y_max, label='Max interp Dice')

    axis.scatter(xx, yy, s=5, label='Sampled points')
    axis.set_yticks(orders)
    axis.set_xlabel('β')
    axis.set_title(title)
    axis.set_ylabel('Polynomial order')
    return im




fig = plt.figure()
plot_dices(fig, fig.add_subplot(221), dices[..., 0], 'CSF')
plot_dices(fig, fig.add_subplot(222), dices[..., 1], 'GM')
plot_dices(fig, fig.add_subplot(223), dices[..., 2], 'WM')
plot_dices(fig, fig.add_subplot(224), dice_mean, 'Mean')
# fig.colorbar(im, ax=fig.axes)
# plt.legend()
plt.tight_layout()


i_max, j_max = np.where(dice_mean == dice_mean.max())
x_max = betas[j_max]
y_max = orders[i_max]
print('Best beta:', x_max)
print('Best order:', y_max)

fig.savefig(figure_path, dpi=400)
print('Showing figure...')
plt.show()
