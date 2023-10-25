import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context("paper", font_scale = 2)
sns.set_style('whitegrid')
#sns.set_style('ticks')

# fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# x = np.arange(4)

# jc4 = [1,3,2,0]
# jc3 = [2,4,2,1]
# jc2 = [6,6,4,0]
# jc1 = [4,8,5,3]

# b4 = ax.bar(x-0.2, jc4)
# b3 = ax.bar(x-0.1, jc3)
# b2 = ax.bar(x+0.1, jc2)
# b1 = ax.bar(x+0.2, jc1)

# ax.bar_label(b1, fontsize=12, rotation=90, padding=3)
# ax.bar_label(b2, fontsize=12, rotation=90, padding=3)
# ax.bar_label(b3, fontsize=12, rotation=90, padding=3)
# ax.bar_label(b4, fontsize=12, rotation=90, padding=3)

# ax.set_xticks(x)
# ax.set_xticklabels(['Antlia', 'Fornax', 'Hydra', 'Control'])
# ax.set_ylabel('No. of candidates')

# ax.legend(['JClass 4', 'JClass 3', 'JClass 2', 'JClass 1'])

# plt.show()


# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np

species = ("Antlia", "Fornax", "Hydra", "Control")
penguin_means = {
    'JClass 4': (1,3,2,0),
    'JClass 3': (2,4,2,1),
    'JClass 2': (6,6,4,0),
    'JClass 1': (4,8,5,3)
}

colors = ['#7f2704', '#d94801', '#fd8d3c', '#fdd0a2']

x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars
multiplier = -0.55
i = 0

fig, ax = plt.subplots(figsize=(12, 6))
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])
    ax.bar_label(rects, padding=3)
    multiplier += 1
    i += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('No. of candidates')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncol=4)
ax.set_ylim(0, 10)

plt.savefig('jclass_bar_plot.png',dpi=250,bbox_inches='tight')

plt.show()
