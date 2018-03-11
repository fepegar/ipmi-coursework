from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

project_dir = Path(__file__).parents[1]
output_path = Path(project_dir, 'BlueRedDivergingAlpha.ctbl')

N = 256
alphas = np.linspace(1, 0, N // 2).tolist()
alphas += list(reversed(alphas))
lines = []
for i, alpha in enumerate(alphas):
    r, g, b, a = plt.cm.RdBu(1 - i / N, alpha=alpha, bytes=True)
    lines.append(f'{i} {i} {r} {g} {b} {int(alpha * 255)}')
lines.append('256 NaN 0 255 0 255')

output_path.write_text('\n'.join(lines))
