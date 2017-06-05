import os, sys

if len(sys.argv)<3:
        print "label_sample.py <generated_image> <robot_view>"

rootdir = sys.argv[1]
wf = open('data_feature/' + sys.argv[2] + ".txt", 'w')


def splited_roobot_view(rootdir):
    count = 0
    for parent,dirnames,filenames in os.walk(rootdir):
        for dirname in  dirnames:
            print "parent is: " + parent
            print  "dirname is: " + dirname

            for _parent,_dirnames,_filenames in os.walk(os.path.join(parent, dirname)):
                for _filename in _filenames:
                    print os.path.join(_parent,_filename), count
                    wf.write(str(os.path.join(_parent, _filename) + ' ' + str(count) + '\n'))
            count = count + 1




