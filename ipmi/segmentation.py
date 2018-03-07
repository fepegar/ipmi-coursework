import numpy as np
import nibabel as nib

# from .path import ensure_dir

# From the workshop
COEF_VARIANCES_1 = 10
COEF_VARIANCES_2 = 0.8

MAX_ITERATIONS = 100


class ExpectationMaximisation:

    def __init__(self, image_path, num_classes, means=None, variances=None,
                 priors=None, epsilon=1e-5):
        self.num_classes = num_classes
        self.image_nii = nib.load(image_path)
        self.image_data = self.image_nii.get_data().astype(float)
        self.epsilon = epsilon

        if means is None:
            means = self.get_random_means(self.image_data, self.num_classes)
        self.means = means

        if variances is None:
            variances = self.get_random_variances(self.image_data,
                                                  self.num_classes)
        self.variances = variances



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


    def gaussian(self, x, variance):
        a = 1 / np.sqrt(2 * np.pi * variance)
        b = np.exp(-x**2 / (2 * variance))
        return a * b


    def umrf(self):
        return


    def run_em(self):
        convergence = False
        y = self.image_data
        K = self.num_classes
        costs = []
        p_shape = list(y.shape) + [self.num_classes]
        p = np.empty(p_shape)
        # mrf = np.ones_like(p)
        old_log_likelihood = -np.inf
        iterations = 0
        np.set_printoptions(precision=2)
        while not convergence:
            print('\n\nIteration', iterations)

            print('\nMeans:')
            print(self.means)

            print('\nVariances:')
            print(self.variances)

            # Expectation
            p_sum = np.zeros_like(y)
            for k in range(K):
                gaussian_pdf = self.gaussian(y - self.means[k],
                                             self.variances[k])
                p[..., k] = gaussian_pdf
                p_sum += p[..., k]

            # Normalise posterior
            for k in range(K):
                p[..., k] /= p_sum + np.spacing(1)

            # Maximization
            for k in range(K):
                self.means[k] = (p[..., k] * y).sum() / p[..., k].sum()
                aux = (y - self.means[k])**2
                self.variances[k] = (p[..., k] * aux).sum() / p[..., k].sum()

            # Cost function
            log_likelihood = np.sum(np.log(p_sum))
            cost = 1 - abs(log_likelihood / old_log_likelihood)
            costs.append(cost)
            print('\nCost:')
            print(cost)

            old_log_likelihood = log_likelihood

            if cost < self.epsilon:
                convergence = True
                print('Algorithm converged with cost', cost)

            iterations += 1
            if iterations > MAX_ITERATIONS:
                print(MAX_ITERATIONS, 'iterations without convergence')
                break

        return p, costs


    def run(self, output_dir):
        import os
        os.makedirs(output_dir)
        p, costs = self.run_em()
        for k in range(self.num_classes):
            nii = nib.Nifti1Image(p[..., k], self.image_nii.affine)
            nib.save(nii, os.path.join(output_dir, f'{k}.nii.gz'))

if __name__ == '__main__':
    em = ExpectationMaximisation('/tmp/em/im.nii.gz', 4)
    em.run('/tmp/em_out')
