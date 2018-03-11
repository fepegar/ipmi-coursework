from collections import namedtuple

import numpy as np
import nibabel as nib
from scipy.ndimage import convolve, generate_binary_structure

from .path import ensure_dir

# From the workshop
COEF_VARIANCES_1 = 10
COEF_VARIANCES_2 = 0.8

MAX_ITERATIONS = 100
EPSILON_STABILITY = np.spacing(1)



def get_brain_mask_from_label_map(label_map_path, brain_mask_path):
    nii = nib.load(str(label_map_path))
    data = nii.get_data()
    brain = (data > 0).astype(np.uint8)
    nii = nib.Nifti1Image(brain, nii.affine)
    ensure_dir(brain_mask_path)
    nib.save(nii, str(brain_mask_path))


def mask(image_path, mask_path, result_path):
    image_nii = nib.load(str(image_path))
    image_data = image_nii.get_data()
    mask_data = nib.load(str(mask_path)).get_data()
    image_data[mask_data == 0] = 0
    result_nii = nib.Nifti1Image(image_data, image_nii.affine)
    nib.save(result_nii, str(result_path))


def dice_score(array1, array2):
    a = array1.astype(bool)
    b = array2.astype(bool)
    TP = (a & b).sum()
    FP = ((a ^ b) & a).sum()
    FN = ((a ^ b) & b).sum()
    score = 2 * TP / (2 * TP + FP + FN)
    return score



class ExpectationMaximisation:

    def __init__(self, image_path, num_classes=None,
                 priors_paths_map=None, beta=0.5, epsilon_convergence=1e-4):
        self.image_nii = nib.load(str(image_path))
        self.image_data = self.image_nii.get_data().astype(float)
        self.epsilon_convergence = epsilon_convergence
        self.beta = beta
        self.priors = None
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


    def read_priors(self, priors_paths_map):
        priors_shape = list(self.image_data.shape) + [len(priors_paths_map)]
        priors = np.empty(priors_shape)
        for k, path in priors_paths_map.items():
            priors[..., k] = nib.load(str(path)).get_data()
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
        for k in range(self.num_classes):
            pik = priors[..., k]
            mean = np.sum(pik * image_data) / np.sum(pik)
            sq_diff = (image_data - mean)**2

            means[k] = mean
            variances[k] = np.sum(pik * sq_diff) / np.sum(pik)
        return means, variances


    def gaussian(self, x, variance):
        a = 1 / np.sqrt(2 * np.pi * variance)
        b = np.exp(-x**2 / (2 * variance))
        return a * b


    def umrf(self, pik, k_class):
        G = -np.eye(self.num_classes) + 1
        umrf = np.zeros_like(pik[..., 0])
        kernel = generate_binary_structure(3, 1)
        kernel[1, 1, 1] = 0
        for j_class in range(self.num_classes):
            umrf += convolve(pik[..., j_class], kernel) * G[k_class, j_class]
        return umrf


    def run_em(self):
        y = self.image_data
        K = self.num_classes
        costs = []
        p_shape = list(y.shape) + [self.num_classes]
        p = np.empty(p_shape)
        old_log_likelihood = -np.inf
        mrf = np.ones_like(p)
        np.set_printoptions(precision=0)

        iterations = 0
        while iterations < MAX_ITERATIONS:
            print('Iteration number', iterations)

            print('\nMeans:', self.means.astype(int))
            print('Variances:', self.variances.astype(int))

            # Expectation
            p_sum = np.zeros_like(y)
            for k in range(K):
                gaussian_pdf = self.gaussian(y - self.means[k],
                                             self.variances[k])
                p[..., k] = gaussian_pdf * mrf[..., k]
                if self.priors is not None:
                    p[..., k] *= self.priors[..., k]
                p_sum += p[..., k]

            # Normalise posterior
            for k in range(K):
                p[..., k] /= p_sum + EPSILON_STABILITY

            # Maximisation
            for k in range(K):
                # Update means
                num = (p[..., k] * y).sum()
                den = p[..., k].sum() + EPSILON_STABILITY
                self.means[k] = num / den

                # Update variances
                aux = (y - self.means[k])**2
                num = (p[..., k] * aux).sum()
                den = p[..., k].sum() + EPSILON_STABILITY
                self.variances[k] = num / den

                # Update MRF
                mrf[..., k] = np.exp(-self.beta * self.umrf(p, k))

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

        np.set_printoptions(precision=8)  # back to default
        Results = namedtuple('EMSegmentationResults', ['probabilities', 'costs'])
        return Results(probabilities=p, costs=costs)


    def write_labels(self, probabilities, segmentation_paths_map):
        for tissue, seg_path in segmentation_paths_map.items():
            binary = (probabilities[..., tissue] > 0.5).astype(np.uint8)
            nii = nib.Nifti1Image(binary, self.image_nii.affine)
            ensure_dir(seg_path)
            nib.save(nii, str(seg_path))


    def run(self, segmentation_paths_map, costs_path=None):
        probabilities, costs = self.run_em()
        self.write_labels(probabilities, segmentation_paths_map)
        if costs_path is not None:
            ensure_dir(costs_path)
            np.save(str(costs_path), costs)
