
from scipy import stats
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

import ipmi

sns.set()
sns.set_context('paper')

subjects = ipmi.get_unsegmented_subjects()
ages = [s.age for s in subjects]

sns.distplot(ages, kde=False, rug=True, fit=stats.norm)
plt.show()
