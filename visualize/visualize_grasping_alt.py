#  Copyright (c) 2024. Jan-Gerrit Habekost. GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.html.
# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

font = {'weight' : 'bold',
        'size'   : 22}

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 11
another_size=12

plt.rc('font', size=another_size)          # controls default text sizes
#plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=another_size)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)



category_names = ("Success", "Slipped", "Not Grasped", "Detection", "LLM")#, "Slipped")#, "Lift")
results = {
    'NICO': (82, 2, 8, 8, 0),#, np.array([84,1 ])),
    'NICOL': (72, 15, 7, 4, 2)#, np.array([87])),
}

#category_names = ['Strongly disagree', 'Disagree',
#                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
#results = {
#    'Question 1': [10, 15, 17, 32, 26],
#    'Question 2': [26, 22, 29, 10, 13],
#    'Question 3': [35, 37, 7, 2, 19],
#    'Question 4': [32, 11, 9, 15, 33],
#    'Question 5': [21, 29, 5, 5, 40],
#    'Question 6': [8, 19, 5, 30, 38]
#}


labels = list(results.keys())
data = np.array(list(results.values()))
data_cum = data.cumsum(axis=1)
#category_colors = plt.colormaps['coolwarm'](
#    np.linspace(0.15, 0.85, data.shape[1]))
category_colors = ['seagreen', 'cornflowerblue', 'firebrick', 'chocolate', 'gold']
fig, ax = plt.subplots(figsize=(9.2, 4))
ax.invert_yaxis()
ax.xaxis.set_visible(True)
ax.set_xlim(0, np.sum(data, axis=1).max())
#ax.set_yticks([0., 1.], labels=labels)
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    widths = data[:, i]
    starts = data_cum[:, i] - widths
    if i < 4:
        rects = ax.barh(labels, widths, left=starts, height=0.75,
                        label=colname, color=color, align='center', alpha=0.75)
    else:
        rects = ax.barh(('NICOL'), 2, left=starts[1], height=0.75,
                        label=colname, color=color, align='center', alpha=0.75)
    r, g, b = mcolors.to_rgb(color)
    text_color = 'white' if i != 4 else 'darkslategray'
    print(rects)
    ax.bar_label(rects, label_type='center', color=text_color)

print(len(category_names))
ax.legend(ncols=len(category_names), bbox_to_anchor=(0.5, 1.20),
          loc='upper center', fontsize=13)
plt.tight_layout()
#survey(results, category_names)
#plt.show()


# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_ylabel('Amount')
#ax.set_title('Grasp')
#species=list(category_names)
#species.append('Slipped')
#species.append('Lift')
#species=tuple(species)
#x = np.arange(len(species))
#ax.set_xticks(x + width, species, weight = 'bold')

#ax.set_ylim(0, 100)

#ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
fig.savefig('./img/grasping_IROS/nico_nicol_grasp_success_alt.png', format='png', dpi=300)
plt.show()