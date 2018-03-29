from collections import namedtuple

import numpy as np
from scipy.sparse import diags
from scipy.ndimage import convolve, generate_binary_structure

from . import nifti
from .path import ensure_dir
from . import constants as const

# From the workshop
COEF_VARIANCES_1 = 10
COEF_VARIANCES_2 = 0.8

MAX_ITERATIONS = 30
EPSILON_STABILITY = np.spacing(1)

np.random.seed(0)  # for reproducibility


def get_brain_mask_from_label_map(label_map_path, brain_mask_path):
    nii = nifti.load(label_map_path)
    data = nii.get_data()
    brain = (data > 0).astype(np.uint8)
    nifti.save(brain, nii.affine, brain_mask_path)


def mask(image_path, mask_path, result_path):
    image_nii = nifti.load(image_path)
    image_data = image_nii.get_data()
    mask_data = nifti.load(mask_path).get_data()
    image_data[mask_data == 0] = 0
    nifti.save(image_data, image_nii.affine, result_path)


def dice_score(array1, array2):
    X = array1.astype(bool)
    Y = array2.astype(bool)
    intersection = np.count_nonzero(X & Y)
    num_X = np.count_nonzero(X)
    num_Y = np.count_nonzero(Y)
    QS = 2 * intersection / (num_X + num_Y)  # "quotient of similarity"
    return QS


def label_map_dice_scores(image_path1, image_path2):
    data1 = nifti.load(image_path1).get_data()
    data2 = nifti.load(image_path2).get_data()
    labels1 = np.unique(data1)
    labels2 = np.unique(data2)
    if not np.all(np.equal(labels1, labels2)):
        raise ValueError(f'{image_path1} and {image_path2} '
                         'have different labels')
    scores = {}
    for label in labels1:
        scores[label] = dice_score(data1 == label, data2 == label)
    return scores


def get_label_map_volumes(image_path):
    nii = nifti.load(image_path)
    voxel_volume = nifti.get_voxel_volume(nii)
    data = nii.get_data()
    volumes = {}
    labels = np.unique(data)
    for label in labels:
        N = np.count_nonzero(data == label)
        total_volume = N * voxel_volume
        volumes[label] = total_volume
    return volumes



class ExpectationMaximisation:

    def __init__(self, image_path, priors_paths_map=None, num_classes=None,
                 write_intermediate=True, segmentation_path=None,
                 use_mrf=True, use_bias_correction=True):
        self.image_nii = nifti.load(image_path)
        self.image_data = self.image_nii.get_data().astype(float)
        self.epsilon_convergence = const.EPSILON_CONVERGENCE_DEFAULT
        self.use_mrf = use_mrf
        self.use_bias_correction = use_bias_correction
        self.beta = const.BETA_DEFAULT
        self.priors = None
        self.write_intermediate = write_intermediate
        self.segmentation_path = segmentation_path

        if priors_paths_map is None:
            if num_classes is None:
                raise ValueError('The number of classes must be specified')
            self.num_classes = num_classes
            self.means = self.get_random_means(self.image_data,
                                               self.num_classes)
            self.variances = self.get_random_variances(self.image_data,
                                                       self.num_classes)
        else:
            self.num_classes = num_classes = len(priors_paths_map)
            self.priors = self.read_priors(priors_paths_map)
            m, v = self.get_means_and_variances_from_priors(self.image_data,
                                                            self.priors)
            self.means, self.variances = m, v


    def set_beta(self, beta):
        self.beta = beta


    def set_convergence_threshold(self, epsilon):
        self.epsilon_convergence = epsilon


    def set_use_mrf(self, use):
        self.use_mrf = use


    def set_use_bias_correction(self, use):
        self.use_bias_correction = use


    def read_priors(self, priors_paths_map):
        priors_shape = list(self.image_data.shape) + [len(priors_paths_map)]
        priors = np.empty(priors_shape)
        for k, path in priors_paths_map.items():
            priors[..., k] = nifti.load(path).get_data()
        return priors


    def get_random_means(self, image_data, num_classes):
        image_min = image_data.min()
        image_max = image_data.max()
        means = np.random.rand(num_classes)
        means *= image_max - image_min
        means += image_min
        return means


    def get_random_variances(self, image_data, num_classes):
        variances = np.random.rand(num_classes)
        variances *= COEF_VARIANCES_1
        variances += COEF_VARIANCES_2 * np.ptp(image_data)
        return variances


    def get_means_and_variances_from_priors(self, image_data, priors):
        means = np.empty(self.num_classes)
        variances = np.empty(self.num_classes)
        weighted_sums = (priors * image_data[..., np.newaxis]).sum(axis=(0, 1, 2))
        sums = priors.sum(axis=(0, 1, 2))
        means = weighted_sums / sums
        sq_diffs = (image_data[..., np.newaxis] - means)**2
        variances = (priors * sq_diffs).sum(axis=(0, 1, 2)) / sums
        return means, variances


    def gaussian(self, x, variance):
        a = 1 / np.sqrt(2 * np.pi * variance)
        b = np.exp(-x**2 / (2 * variance))
        return a * b


    def u_mrf(self, pik, k_class):
        # MRF energy function
        G = -np.eye(self.num_classes) + 1
        u_mrf = np.zeros_like(pik[..., 0])

        # The kernel is a 3D Greek cross...
        kernel = generate_binary_structure(rank=3, connectivity=1)
        # ...with an empty center
        kernel[1, 1, 1] = 0

        # This loop can probably be avoided with a 4D kernel
        for j_class in range(self.num_classes):
            u_mrf += convolve(pik[..., j_class], kernel) * G[k_class, j_class]
        return u_mrf


    def get_bases_matrix(self, image, order):
        grid = np.meshgrid(*map(range, image.shape))

        bases = []
        bases.append(lambda x, y, z: np.ones_like(x))
        if order >= 1:
            bases.append(lambda x, y, z: x)
            bases.append(lambda x, y, z: y)
            bases.append(lambda x, y, z: z)
        if order >= 2:
            bases.append(lambda x, y, z: x**2)
            bases.append(lambda x, y, z: y**2)
            bases.append(lambda x, y, z: z**2)
            bases.append(lambda x, y, z: x * y)
            bases.append(lambda x, y, z: x * z)
            bases.append(lambda x, y, z: y * z)
        if order >= 3:
            bases.append(lambda x, y, z: x**3)
            bases.append(lambda x, y, z: y**3)
            bases.append(lambda x, y, z: z**3)
            bases.append(lambda x, y, z: x**2 * y)
            bases.append(lambda x, y, z: x**2 * z)
            bases.append(lambda x, y, z: y**2 * x)
            bases.append(lambda x, y, z: y**2 * z)
            bases.append(lambda x, y, z: z**2 * x)
            bases.append(lambda x, y, z: z**2 * y)
            bases.append(lambda x, y, z: x * y * z)

        columns = np.column_stack([basis(*grid).ravel() for basis in bases])
        return columns  # N x M


    def get_bias_field(self, probabilities, A):
        y = self.image_data  # si x sj x sk
        weights = probabilities / self.variances  # si x sj x sk x 4
        w_i = weights.sum(axis=3)  # si x sj x sk
        W = diags(w_i.ravel())  # N x N (sparse)
        num = (weights * self.means).sum(axis=3)
        den = weights.sum(axis=3) + EPSILON_STABILITY
        y_tilde = num / den  # predicted signal, si x sj x sk
        At = A.T  # M x N
        R = (y - y_tilde).reshape(-1, 1)  # N x 1
        WR = W.dot(R)  # NxN x Nx1 = N x 1
        AtWR = np.matmul(At, WR)  # MxN x Nx1 = M x 1
        WA = W.dot(A)  # NxN x NxM = N x M
        AtWA = np.matmul(At, WA)  # MxN x NxM = M x M
        C = np.matmul(np.linalg.inv(AtWA), AtWR)  # MxM x Mx1 = M x 1
        BF = np.matmul(A, C).reshape(y.shape)  # sum_k(c_k phi_k(x_i))
        return BF


    def run_em(self):
        y = self.image_data
        K = self.num_classes
        costs = []
        p_shape = list(y.shape) + [self.num_classes]
        p = np.empty(p_shape)
        old_log_likelihood = -np.inf
        mrf = np.ones_like(p)
        if self.use_bias_correction:
            A = self.get_bases_matrix(y, 2)  # N x M
        BF = np.zeros_like(y)

        np.set_printoptions(precision=0)

        if self.write_intermediate and self.priors is not None:
            print('Writing initial segmentation (priors)...')
            segmentation_path = str(self.segmentation_path).replace(
                '.nii.gz', '_iteration_0_priors.nii.gz')
            self.write_labels(self.priors, segmentation_path)

        iterations = 0
        while iterations < MAX_ITERATIONS:
            print('Iteration number', iterations + 1)
            print('\nMeans:', self.means.astype(int))
            print('Variances:', self.variances.astype(int))


            ## Expectation ##
            print('\n-- Expectation --')
            print('Classifying...')

            # Eq (1) of Van Leemput 1
            p = self.gaussian(
                y[..., np.newaxis] - self.means - BF[..., np.newaxis],
                self.variances)
            if self.priors is not None:
                p *= self.priors
            p *= mrf
            p_sum = p.sum(axis=3)

            # Normalise posterior (Eq (2) of Van Leemput 1)
            p /= p.sum(axis=3)[..., np.newaxis] + EPSILON_STABILITY

            if self.write_intermediate:
                print('Writing intermediate results...')
                segmentation_path = str(self.segmentation_path).replace(
                    '.nii.gz', f'_iter_{iterations+1}.nii.gz')
                self.write_labels(p, segmentation_path)


            ## Maximisation ##
            # "class distribution parameter estimation"

            # Update means (Eq (3) of Van Leemput 1)
            print('\n-- Maximisation --')
            print('Updating means...')
            y_unbiased = y - BF
            num = (p * y_unbiased[..., np.newaxis]).sum(axis=(0, 1, 2))
            den = p.sum(axis=(0, 1, 2)) + EPSILON_STABILITY
            self.means = num / den

            # Update variances (Eq (4) of Van Leemput 1)
            print('Updating variances...')
            sq_diffs = (y_unbiased[..., np.newaxis] - self.means)**2
            num = (p * sq_diffs).sum(axis=(0, 1, 2))
            den = p.sum(axis=(0, 1, 2)) + EPSILON_STABILITY
            self.variances = num / den

            # Update MRF
            if self.use_mrf:
                print('Updating MRF...')
                for j in range(K):
                    mrf[..., j] = np.exp(-self.beta * self.u_mrf(p, j))

            # Intensity non-uniformity correction #
            if self.use_bias_correction:
                print('Updating INU correction coefficients...')
                BF = self.get_bias_field(p, A)


            # Cost function
            log_likelihood = np.sum(np.log(p_sum + EPSILON_STABILITY))
            print('\n{:20}'.format('Old log likelihood:'), old_log_likelihood)
            print('{:20}'.format('Log likelihood:'), log_likelihood)
            print('{:20}'.format('New / old:'),
                  abs(log_likelihood / old_log_likelihood))
            print('{:20}'.format('New - old:'),
                  abs(log_likelihood - old_log_likelihood))

            cost = 1 - abs(log_likelihood / old_log_likelihood)
            costs.append(cost)
            print('Cost:', cost)

            old_log_likelihood = log_likelihood

            # We don't want to stop when cost < 0
            if cost >= 0 and cost < self.epsilon_convergence:
                print('Algorithm converged with cost', cost)
                break

            iterations += 1

            print(2 * '\n')
            print(50 * '*')
            print(2 * '\n')
        else:
            print(MAX_ITERATIONS, 'iterations reached without convergence')

        print(5 * '\n')
        np.set_printoptions(precision=8)  # back to default
        Results = namedtuple('EMSegmentationResults',
                             ['probabilities', 'costs'])
        if self.write_intermediate:
            bias_field_path = str(self.segmentation_path).replace(
                '.nii.gz', f'_bias_field.nii.gz')
            nifti.save(BF, self.image_nii.affine, bias_field_path)

        return Results(probabilities=p, costs=costs)


    def write_labels(self, probabilities, segmentation_path):
        label_map = np.zeros_like(probabilities[..., 0], np.uint8)
        for tissue in range(self.num_classes):
            mask_data = probabilities[..., tissue] > 0.5
            label_map[mask_data] = tissue
        nifti.save(label_map, self.image_nii.affine, segmentation_path)


    def run(self, segmentation_path, costs_path=None, costs_plot_path=None):
        self.segmentation_path = segmentation_path
        results = self.run_em()
        self.write_labels(results.probabilities, segmentation_path)
        if costs_path is not None:
            ensure_dir(costs_path)
            np.save(str(costs_path), results.costs)

        if costs_plot_path is not None:
            import matplotlib as mpl
            mpl.use('TkAgg')
            from matplotlib.pyplot import figure
            fig = figure()
            axis = fig.gca()
            axis.set_title('Cost vs iterations')
            axis.set_xlabel('Iterations')
            axis.set_ylabel('Cost')
            axis.grid(True)
            axis.set_yscale('log')
            axis.plot(results.costs, '-o')
            axis.xticks(range(0, len(results.costs) + 1, 2))
            fig.savefig(costs_plot_path, dpi=400)
        return results
