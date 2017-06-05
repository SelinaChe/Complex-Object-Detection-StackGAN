# -*- coding: utf-8 -*-
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import Image
import sys,os
import numpy as np

def set_image_label(_image_path):
    label_file = open('data_feature/%s_test.txt' % gan_image.split('.')[0], 'w')#open(os.path.join('feature_data', os.path.basename(image_path)+'_label.txt'), 'w')
    print 'data_feature/%s_test.txt' % gan_image.split('.')[0]
    label_file.write(_image_path+' '+str(0)+'\n')

def set_cropImage_label(rootdir):
    count = 0
    print 'data_feature/%s_test.txt' % image_path.split('.')[0]
    wf = open('data_feature/%s_train.txt' % image_path.split('.')[0], 'w')
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            #print os.path.join(parent,filename), count
            wf.write(str(os.path.join(parent, filename) + ' ' + filename.split('_')[0] + '\n'))
            count = count + 1
'''
    for parent,dirnames,filenames in os.walk(rootdir):
        for dirname in  dirnames:
            print "parent is: " + parent
            print  "dirname is: " + dirname

            for _parent,_dirnames,_filenames in os.walk(os.path.join(parent, dirname)):
                for _filename in _filenames:
                    print os.path.join(_parent,_filename), count
                    wf.write(str(os.path.join(_parent, _filename) + ' ' + str(count) + '\n'))
            count = count + 1
'''
image_path =  sys.argv[1]
gan_image = sys.argv[2]

def main():

    # loading astronaut image
    #img = skimage.data.astronaut() 
#    image_path =  sys.argv[1]
#    gan_image = sys.argv[2]
    image = Image.open(image_path)
    img = np.array(image)
    #image = Image.open('')

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    #ax.imshow(img)
    count = 0
    for x, y, w, h in candidates:
        #print x, y, w, h
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    #    ax.add_patch(rect)
        region = (x,y,x+w,y+h)
        cropImg = image.crop(region)
        sub_image_folder = os.path.basename(image_path).split('.')[0]#.split('.')[0].split('/')[-1]
        #print sub_image_folder
        sub_path = os.getcwd() + '/region/' + sub_image_folder
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        #cropImg.save(str(count)+'.jpg')
        #count = count + 1
        cropImg.save(r'%s/%d_%d_%d_%d_%d.jpg'%(sub_path, count, x, y, w, h))
	count = count + 1
        #plt.show(rect)
    
    #gan_image = sys.argv[2]
    set_image_label(gan_image)
    set_cropImage_label(sub_path)
    

    #plt.show()

if __name__ == "__main__":
    print("# Usage: example.py <robot_view> <generated view>\n")
    main()
