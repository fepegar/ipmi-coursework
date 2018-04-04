import pathlib
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from ipmi.path import dice_report_pandas_path

repo_dir = pathlib.Path(__file__).parents[2]
figure_path = repo_dir / 'latex' / 'figures' / 'dices_bars.png'

sns.set()
sns.set_context('paper')

dices = pandas.read_csv(dice_report_pandas_path)
grid = sns.factorplot(x='Subject ID', y='Dice score', data=dices, hue='Tissue',
                      kind='bar')

grid.ax.set_ybound(lower=0.7)

fig = plt.gcf()
fig.set_size_inches(8, 4)
fig.savefig(figure_path, dpi=400)
