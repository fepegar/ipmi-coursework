import numpy as np
import nibabel as nib

# from .path import ensure_dir

# From the workshop
COEF_VARIANCES_1 = 10
COEF_VARIANCES_2 = 0.8

MAX_ITERATIONS = 100
EPSILON_STABILITY = np.spacing(1)


class ExpectationMaximisation:

    def __init__(self, image_path, num_classes=None,
                 priors_paths_map=None, epsilon_convergence=1e-5):
        self.image_nii = nib.load(image_path)
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
            self.means = self.get_means_from_priors(self.image_data,
                                                    self.priors)
            self.variances = self.get_variances_from_priors(self.image_data,
                                                            self.priors)


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


    def get_means_from_priors(self, image_data, priors):
        return self.get_random_means(image_data, priors.shape[-1])


    def get_variances_from_priors(self, image_data, priors):
        return self.get_random_variances(image_data, priors.shape[-1])


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
        old_log_likelihood = -np.inf
        iterations = 0
        np.set_printoptions(precision=2)
        while not convergence or iterations > MAX_ITERATIONS:
            print('\n\n\n\nIteration', iterations)

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

            # Cost function
            log_likelihood = np.sum(np.log(p_sum + EPSILON_STABILITY))
            cost = 1 - abs(log_likelihood / old_log_likelihood)
            costs.append(cost)
            print('\nCost:')
            print(cost)

            old_log_likelihood = log_likelihood

            if cost < self.epsilon_convergence:
                convergence = True
                print('Algorithm converged with cost', cost)

            iterations += 1
            if iterations > MAX_ITERATIONS:
                print(MAX_ITERATIONS, 'iterations without convergence')

        return p, costs


    def write_labels(self, probabilities, output_dir):
        for k in range(self.num_classes):
            nii = nib.Nifti1Image(probabilities[..., k], self.image_nii.affine)
            output_path = output_dir / f'{k}.nii.gz'
            nib.save(nii, str(output_path))


    def run(self, output_dir, costs_path=None):
        probabilities, costs = self.run_em()
        self.write_labels(probabilities, output_dir)
        if costs_path is not None:
            np.save(str(costs_path), costs)
