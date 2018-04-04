import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_context("paper")
# sns.set(color_codes=True)
sns.set()
import ipmi


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

subject = ipmi.get_segmented_subjects()[0]
data = ipmi.nifti.load(subject.t1_path).get_data()
ax = sns.distplot(data.ravel(), kde=False)
means = 25, 24, 46, 72
variances = 844, 88, 78, 34
gains = 200_000, 200_000, 250_000, 250_000
labels = 'BG', 'CSF', 'GM', 'WM'

x = np.arange(256)
for mu, variance, A, label in zip(means, variances, gains, labels):
    sig = np.sqrt(variance)
    ax.plot(x, A * gaussian(x, mu, sig), label=label)
ax.set_ybound(upper=500_000)

plt.legend()
plt.show()
