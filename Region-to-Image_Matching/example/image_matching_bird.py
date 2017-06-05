import os, sys
import numpy as np
import math

print "Usage:	image_matching.py <train_feature> <train_label> <test_feature>"

train_fc8 = np.loadtxt(sys.argv[1])
train_labels = np.loadtxt(sys.argv[2])

test_fc8 = np.loadtxt(sys.argv[3])
#test_labels = np.loadtxt('feature_extraction/test_labels.txt')

def sort_by_value(value):
    items = value.items()
    backitems = [[v[1],v[0]] for v in items]
    backitems.sort()
    return [backitems[i][1] for i in range(0,len(backitems))]

def get_top_number(sorted_dic):
    count = 0
    
    match_file = open('match_result/match_result_%s.txt'% sys.argv[4].split(".")[0].split("/")[-1], 'w')
#    print sorted_dic
    for i in sorted_dic:
        count = count + 1
        #print str(i).split(',')[0].split('(')[1]
#        print "match_result/" + str(i).split(',')[1].split(')')[0]+' '+str(train_labels[int(str(i).split(',')[0].split('(')[1])])
        #print i, train_labels[int(str(i).split(',')[0].split('(')[1])]
        print str(i[1])+' '+str(train_labels[i[0]])
        match_file.write(str(i[1])+' '+str(train_labels[i[0]])+'\n')

        #match_file.write(str(i).split(',')[1].split(')')[0]+' '+str(train_labels[int(str(i).split(',')[0].split('(')[1])])+'\n')
        #if count == 10:
        #    print '*************'
        #    break

'''
#euclidean_distance
euclidean_distance = {}
#print "*************test_fc8****************"
#print len(test_fc8)
#for i in range(len(test_fc8)):

for j in range(len(train_fc8)):
    difference = train_fc8[j] - test_fc8
    euclidean = 0
    for one in difference:
        euclidean = euclidean + one*one
    euclidean_distance[j] = math.sqrt(euclidean)
sorted_euclidean_distance =  sorted(euclidean_distance.items(), key=lambda d: d[1])
get_top_number(sorted_euclidean_distance)
'''


#Hanmin distance
hanmin_distance = {}
#for i in range(len(test_fc8)):
print len(test_fc8)
for one in range(len(test_fc8)):
    if float(test_fc8[one]) < 0.5:
        test_fc8[one] = 0
    else:
        test_fc8[one] = 1
for j in range(len(train_fc8)):
    for one in range(len(train_fc8[j])):
        if float(train_fc8[j][one]) < 0.0:
            train_fc8[j][one] = 0
        else:
            train_fc8[j][one] = 1
    #difference = train_fc8[j] - test_fc8[i]
    hanmin = 0
    for ii in range(len(train_fc8[j])):
        if train_fc8[j][ii] != test_fc8[ii]:
            hanmin = hanmin + 1
    hanmin_distance[j] = hanmin
sorted_hanmin_distance = sorted(hanmin_distance.items(), key=lambda d: d[1])
get_top_number(sorted_hanmin_distance)

