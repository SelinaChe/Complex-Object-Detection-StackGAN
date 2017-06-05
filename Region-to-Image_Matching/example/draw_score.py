import os, sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Image
import numpy as np

cls = sys.argv[1]

image_path =  'flower/%s.jpg' % cls
image = Image.open(image_path)
img = np.array(image)

match_read = open('match_result/match_result_%s.txt' % cls, 'r')
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(40, 40))
ax.imshow(img)

match_scores = match_read.read().split('\n')
match_scores_reverse = []
for i in range(len(match_scores)):
    match_scores_reverse.append(match_scores[len(match_scores)-i-1])

for line in match_scores_reverse:
    if line != '':
        distance = float(line.split()[0])
        label = float(line.split()[1])
        #score = 2*(1 / (1 + math.exp(-1*(1 / (math.log((distance-50),2)))**2))-0.5)
        #if distance == 9.0:
        #    distance = 10.0
        #print distance
        #while distance <= 9.0:
        #    distance = distance + 1.0
        score = 2*(1 / (1 + math.exp(-14.8*(1 / (distance-10.0))))-0.5)
        print score
        if score >= 0.9: 
            facecolor = "black"
            color = "yellow"
            edgecolor='yellow'
        else:
            facecolor = "blue"
            color = "white"
            edgecolor='red'

        
        for parent,dirnames,filenames in os.walk('region/%s' % cls):
            for filename in filenames:
                if int(label)==int(filename.split('_')[0]):
                    x = int(filename.split('_')[1])
                    y = int(filename.split('_')[2])
                    w = int(filename.split('_')[3])
                    h = int(filename.split('_')[4].split('.')[0])
                    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=edgecolor, linewidth=16)
                    ax.add_patch(rect)
                    ax.text(x,y-2, '{:.4f}'.format(score), bbox=dict(facecolor=facecolor, alpha=0.5), fontsize=100, color=color)
                    #region = (x,y,x+w,y+h)
#plt.show()
plt.axis('off')
plt.draw()

fig = plt.gcf()
fig.savefig("match_images/macthed_%s.jpg" % cls)
#plt.show()
