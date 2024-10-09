#  Copyright (c) 2024. Jan-Gerrit Habekost. GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.html.
# data from https://allisonhorst.github.io/palmerpenguins/

import matplotlib.pyplot as plt
import numpy as np


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
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=another_size)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)


species = ("Success", "Missed", "Detection", "LLM")#, "Slipped")#, "Lift")
penguin_means = {
    'NICO': (82, 8, 8, 0),#, np.array([84,1 ])),
    'NICOL': (72, 7, 4, 2)#, np.array([87])),
}

slipped_nico = (2)
slipped_nicol = (15)

upper = (2, 15 )
lower = (82, 72)

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0.5

fig, ax = plt.subplots(layout='constrained')
color = ['cornflowerblue', 'sandybrown']

even=True

count=0
full_offset = 0
for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    color_index = 0 if even else 1
    print(attribute)
    print(measurement)

    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color[color_index], edgecolor="black")
    #rects[4].linestyle='--'
    if even:
        even = False
    else:
        even = True
    ax.bar_label(rects, padding=3)
    multiplier += 1
    full_offset = width * multiplier
    count +=1

ax.legend(loc='upper right', ncols=1, bbox_to_anchor=(0.83, 1.0))

rects = ax.bar(3.5 + full_offset, slipped_nico, width,  color=color[0], edgecolor="black",linestyle='--', alpha=0.8)
ax.bar_label(rects, padding=3)
rects = ax.bar(3.75 + full_offset, slipped_nicol, width,color=color[1], edgecolor="black",linestyle='--', alpha=0.8)
ax.bar_label(rects, padding=3)

rects = ax.bar(4.5 + full_offset, lower[0], width,  color=color[0], edgecolor="black",bottom=[0] )
ax.bar_label(rects, padding=-15)
rects = ax.bar(4.5 + full_offset, upper[0], width,color=color[0], edgecolor="black",bottom=lower[0] ,linestyle='--', alpha=0.8)
ax.bar_label(rects, padding=3)
rects = ax.bar(4.75 + full_offset, lower[1], width,  color=color[1], edgecolor="black",bottom=[0] )
ax.bar_label(rects, padding=-15)
rects = ax.bar(4.75 + full_offset, upper[1], width,  color=color[1], edgecolor="black",bottom=lower[1],linestyle='--' , alpha=0.8)
ax.bar_label(rects, padding=3)
ax.vlines(x=[4.75], ymin=0, ymax=100, color='black', label='test lines', ls='--')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Amount')
#ax.set_title('Grasp')
species=list(species)
species.append('Slipped')
species.append('Lift')
species=tuple(species)
x = np.arange(len(species))
ax.set_xticks(x + width, species, weight = 'bold')

ax.set_ylim(0, 100)

#ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
fig.savefig('./img/grasping_IROS/nico_nicol_grasp_success.png', format='png', dpi=300)
plt.show()