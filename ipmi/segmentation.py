import numpy as np
import nibabel as nib

from .path import ensure_dir

# From the workshop
COEF_VARIANCES_1 = 10
COEF_VARIANCES_2 = 0.8

MAX_ITERATIONS = 100
EPSILON_STABILITY = np.spacing(1)

np.set_printoptions(precision=0)


class ExpectationMaximisation:

    def __init__(self, image_path, num_classes=None,
                 priors_paths_map=None, epsilon_convergence=1e-5):
        self.image_nii = nib.load(str(image_path))
        self.image_data = self.image_nii.get_data().astype(float)
        self.epsilon_convergence = epsilon_convergence
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
        pik_class = pik[..., 0]
        umrf = np.zeros_like(pik_class)
        si, sj, sk = pik_class.shape

        # Comments in RAS wlg
        for i_idx in range(si):
            print(i_idx, si)
            for j_idx in range(sj):
                for k_idx in range(sk):
                    umrf_here = 0
                    for j_class in range(self.num_classes):
                        umrfj = 0

                        # umrf R
                        if (i_idx + 1) < si:
                            umrfj += pik[i_idx + 1, j_idx, k_idx, j_class]

                        # umrf L
                        if (i_idx - 1) >= 0:
                            umrfj += pik[i_idx - 1, j_idx, k_idx, j_class]


                        # umrf A
                        if (j_idx + 1) < sj:
                            umrfj += pik[i_idx, j_idx + 1, k_idx, j_class]

                        # umrf P
                        if (j_idx - 1) >= 0:
                            umrfj += pik[i_idx, j_idx - 1, k_idx, j_class]


                        # umrf S
                        if (k_idx + 1) < sk:
                            umrfj += pik[i_idx, j_idx, k_idx + 1, j_class]

                        # umrf I
                        if (k_idx - 1) >= 0:
                            umrfj += pik[i_idx, j_idx, k_idx - 1, j_class]

                        umrf_here += umrfj * G[k_class, j_class]
                    umrf[i_idx, j_idx, k_idx] = umrf_here
        return umrf


    def run_em(self):
        convergence = False
        y = self.image_data
        K = self.num_classes
        costs = []
        p_shape = list(y.shape) + [self.num_classes]
        p = np.empty(p_shape)
        old_log_likelihood = -np.inf
        iterations = 0
        mrf = np.ones_like(p)
        beta = 2

        while not convergence or iterations > MAX_ITERATIONS:
            print('Iteration number', iterations)

            print('\nMeans:', self.means.astype(int))
            print('\nVariances:', self.variances.astype(int))

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
                num = (p[..., k] * y).sum()
                den = p[..., k].sum() + EPSILON_STABILITY
                self.means[k] = num / den

                aux = (y - self.means[k])**2
                num = (p[..., k] * aux).sum()
                den = p[..., k].sum() + EPSILON_STABILITY
                self.variances[k] = num / den

                try:
                    mrf[..., k] = np.exp(-beta * self.umrf(p, k))
                except KeyboardInterrupt:
                    print('MRF computation interrupted')
                    return p, costs

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
            if cost > 0 and cost < self.epsilon_convergence:
                convergence = True
                print('Algorithm converged with cost', cost)

            iterations += 1
            if iterations > MAX_ITERATIONS:
                print(MAX_ITERATIONS, 'iterations without convergence')

            print(4 * '\n')

        return p, costs


    def write_labels(self, probabilities, output_dir):
        for k in range(self.num_classes):
            nii = nib.Nifti1Image(probabilities[..., k], self.image_nii.affine)
            output_path = output_dir / f'{k}.nii.gz'
            ensure_dir(output_path)
            nib.save(nii, str(output_path))


    def run(self, output_dir, costs_path=None):
        probabilities, costs = self.run_em()
        self.write_labels(probabilities, output_dir)
        if costs_path is not None:
            ensure_dir(costs_path)
            np.save(str(costs_path), costs)
