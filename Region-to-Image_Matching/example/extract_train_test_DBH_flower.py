import time
sttime= time.clock()
print("Importing essential modules...")
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import svm
import sys

caffe_root = "/root/py-faster-rcnn/caffe-fast-rcnn/"  
import sys
#sys.path.insert(0, caffe_root + "python")

import caffe
print("\t\tsuccessfully imported in "+str(time.clock()- sttime)+" Secs.\n")

####################
##   Parameters   ##
####################

#model net to extract features
mynet= "finetuned"        #accepted values are alexnet, caffenet, googlenet
#deploy_cub200.prototxt  deploy_flowers.prototxt  model_birds_ft_iter_10000.caffemodel  model_flowers_ft_iter_10000.caffemodel
if mynet== "finetuned":
    deploy= "/root/selectivesearch/example/cache_iccv_flowers/cnn_finetuning/caffe_reference/deploy.prototxt"
    model= "/root/selectivesearch/example/cache_iccv_flowers/cnn_finetuning/caffe_reference/model_ft_iter_10000.caffemodel"
elif mynet== "caffenet":
    deploy= "/root/py-faster-rcnn/caffe-fast-rcnn/models/bvlc_reference_caffenet/deploy.prototxt"
    model= "/root/py-faster-rcnn/caffe-fast-rcnn/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
elif mynet== "googlenet":
    deploy= "/root/py-faster-rcnn/caffe-fast-rcnn/models/bvlc_googlenet/deploy.prototxt"
    model= "/root/py-faster-rcnn/caffe-fast-rcnn/models/bvlc_googlenet/bvlc_googlenet.caffemodel"
else:
    print("Please give a proper net to extract!")


#########################
##   Global settings   ##
#########################


plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

caffe.set_mode_cpu()
net = caffe.Net(deploy,
                model,
                caffe.TEST)

########################
##   Pre-Processing   ##
########################

transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
transformer.set_transpose("data", (2,0,1))
transformer.set_mean("data", np.load(caffe_root + "python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1)) # mean pixel
transformer.set_raw_scale("data", 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap("data", (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs["data"].reshape(1,3,227,227)

train_feats_fc7= []
test_feats_fc7= []

train_feats_fc8= []
test_feats_fc8= []

train_feats_fc7_fc8= []
test_feats_fc7_fc8= []

#####################
##   Load Labels   ##
#####################

train_labels= []
test_labels= []

print("Loading labels from files...")

base_data = sys.argv[1]
search_data = sys.argv[2]

rdfile1= open(base_data, "r")
rdfile2= open(search_data, "r")

sttime= time.clock()
#to load train_labels from file
for line in rdfile1:
    line= line[:-1]
    train_labels.append(int(line.split()[1]))

train_labels= np.array(train_labels)
np.savetxt(base_data.split('.')[0]+'_label.txt',train_labels)
print("\t\t"+ str(len(train_labels))+" train_labels successfully loaded in "+str(time.clock()- sttime)+" Secs.\n")

sttime= time.clock()
#to load test labels from file
for line in rdfile2:
    line= line[:-1]
    test_labels.append(int(line.split()[1]))

test_labels= np.array(test_labels)
print("\t\t"+ str(len(test_labels))+" test_labels successfully loaded in "+str(time.clock()- sttime)+" Secs.\n")


##########################
##   Extract Features   ##
##########################

#train features
print("Extracting train features...")
sttime= time.clock()
train_file= open(base_data,"r")

total_train_images= str(len(train_labels))
current_image= 1
for line in train_file:
    temp_print= (str(current_image)+ "/"+ total_train_images+ "... ")
    sys.stdout.write(temp_print)
    sys.stdout.flush()
    current_image+= 1

    line1= line.split()
    net.blobs["data"].data[...]= transformer.preprocess("data", caffe.io.load_image(line1[0]))
    out= net.forward()   

    feat_fc7= np.array(net.blobs["fc7"].data[0])
    train_feats_fc7.append(feat_fc7)
    
train_feats_fc7= np.array(train_feats_fc7)
np.savetxt(base_data.split('.')[0]+'_feature.txt', train_feats_fc7)
print("\n\t\tsuccessfully extracted in "+str(time.clock()- sttime)+" Secs.")


#test features
print("Extracting test features...")
sttime= time.clock()
test_file= open(search_data, "r")

total_test_images= str(len(test_labels))
current_image= 1
for line in test_file:
    temp_print= (str(current_image)+ "/"+ total_test_images+ "... ")
    sys.stdout.write(temp_print)
    sys.stdout.flush()
    current_image+= 1

    line1= line.split()
    net.blobs["data"].data[...]= transformer.preprocess("data", caffe.io.load_image(line1[0]))
    out= net.forward()   

    feat_fc7= np.array(net.blobs["fc7"].data[0])
    test_feats_fc7.append(feat_fc7)

test_feats_fc7= np.array(test_feats_fc7)
np.savetxt(search_data.split('.')[0]+'_feature.txt', test_feats_fc7)

print("\n\t\tsuccessfully extracted in "+str(time.clock()- sttime)+" Secs.")


######################
##   SVM Training   ##
######################

'''
#1st svm fit
print("Svm Classification with first 500 features of layer fc7")
C=1.0

print("\tfitting svc...")
sttime= time.clock()
svc = svm.SVC(kernel='linear', C=C).fit(train_feats_fc7[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting rbf_svc...")
sttime= time.clock()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_feats_fc7[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting poly_svc...")
sttime= time.clock()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_feats_fc7[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting lin_svc...")
sttime= time.clock()
lin_svc = svm.LinearSVC(C=C).fit(train_feats_fc7[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\n\nPredicting values...")
for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    sttime= time.clock()
    print("\nClassifier: ")
    print(clf)
    Z = clf.predict(test_feats_fc7[:,:500])
    #print(Z)
    #print(test_labels)
    scores = (Z== test_labels)
    total = np.sum(scores)
    print("Accuracy: ")
    print("\t\t"+str(total/500.0))
    print("Time: ")
    print("\t\t"+ str(time.clock()- sttime)+ " Secs.\n")


#2nd svm fit
print("Svm Classification with first half (2048) features of layer fc7")
C=1.0

print("\tfitting svc...")
sttime= time.clock()
svc = svm.SVC(kernel='linear', C=C).fit(train_feats_fc7[:,:2048], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting rbf_svc...")
sttime= time.clock()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_feats_fc7[:,:2048], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting poly_svc...")
sttime= time.clock()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_feats_fc7[:,:2048], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting lin_svc...")
sttime= time.clock()
lin_svc = svm.LinearSVC(C=C).fit(train_feats_fc7[:,:2048], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\n\nPredicting values...")
for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    sttime= time.clock()
    print("\nClassifier: ")
    print(clf)
    Z = clf.predict(test_feats_fc7[:,:2048])
    #print(Z)
    #print(test_labels)
    scores = (Z== test_labels)
    total = np.sum(scores)
    print("Accuracy: ")
    print("\t\t"+str(total/500.0))
    print("Time: ")
    print("\t\t"+ str(time.clock()- sttime)+ " Secs.\n")


#3rd svm fit
print("Svm Classification with all the features of layer fc7")
C=1.0

print("\tfitting svc...")
sttime= time.clock()
svc = svm.SVC(kernel='linear', C=C).fit(train_feats_fc7, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting rbf_svc...")
sttime= time.clock()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_feats_fc7, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting poly_svc...")
sttime= time.clock()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_feats_fc7, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting lin_svc...")
sttime= time.clock()
lin_svc = svm.LinearSVC(C=C).fit(train_feats_fc7, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\n\nPredicting values...")
for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    sttime= time.clock()
    print("\nClassifier: ")
    print(clf)
    Z = clf.predict(test_feats_fc7)
    #print(Z)
    #print(test_labels)
    scores = (Z== test_labels)
    total = np.sum(scores)
    print("Accuracy: ")
    print("\t\t"+str(total/500.0))
    print("Time: ")
    print("\t\t"+ str(time.clock()- sttime)+ " Secs.\n")


#4th svm fit
print("Svm Classification with first half (500) features of layer fc8")
C=1.0

print("\tfitting svc...")
sttime= time.clock()
svc = svm.SVC(kernel='linear', C=C).fit(train_feats_fc8[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting rbf_svc...")
sttime= time.clock()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_feats_fc8[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting poly_svc...")
sttime= time.clock()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_feats_fc8[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting lin_svc...")
sttime= time.clock()
lin_svc = svm.LinearSVC(C=C).fit(train_feats_fc8[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\n\nPredicting values...")
for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    sttime= time.clock()
    print("\nClassifier: ")
    print(clf)
    Z = clf.predict(test_feats_fc8[:,:500])
    #print(Z)
    #print(test_labels)
    scores = (Z== test_labels)
    total = np.sum(scores)
    print("Accuracy: ")
    print("\t\t"+str(total/500.0))
    print("Time: ")
    print("\t\t"+ str(time.clock()- sttime)+ " Secs.\n")


#5th svm fit
print("Svm Classification with all the features of layer fc8")
C=1.0

print("\tfitting svc...")
sttime= time.clock()
svc = svm.SVC(kernel='linear', C=C).fit(train_feats_fc8, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting rbf_svc...")
sttime= time.clock()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_feats_fc8, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting poly_svc...")
sttime= time.clock()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_feats_fc8, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting lin_svc...")
sttime= time.clock()
lin_svc = svm.LinearSVC(C=C).fit(train_feats_fc8, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\n\nPredicting values...")
for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    sttime= time.clock()
    print("\nClassifier: ")
    print(clf)
    Z = clf.predict(test_feats_fc8)
    #print(Z)
    #print(test_labels)
    scores = (Z== test_labels)
    total = np.sum(scores)
    print("Accuracy: ")
    print("\t\t"+str(total/500.0))
    print("Time: ")
    print("\t\t"+ str(time.clock()- sttime)+ " Secs.\n")


#6th svm fit
print("Svm Classification with all the features of layer fc7_fc8")
C=1.0

print("\tfitting svc...")
sttime= time.clock()
svc = svm.SVC(kernel='linear', C=C).fit(train_feats_fc7_fc8, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting rbf_svc...")
sttime= time.clock()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_feats_fc7_fc8, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting poly_svc...")
sttime= time.clock()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_feats_fc7_fc8, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting lin_svc...")
sttime= time.clock()
lin_svc = svm.LinearSVC(C=C).fit(train_feats_fc7_fc8, train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\n\nPredicting values...")
for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    sttime= time.clock()
    print("\nClassifier: ")
    print(clf)
    Z = clf.predict(test_feats_fc7_fc8)
    #print(Z)
    #print(test_labels)
    scores = (Z== test_labels)
    total = np.sum(scores)
    print("Accuracy: ")
    print("\t\t"+str(total/500.0))
    print("Time: ")
    print("\t\t"+ str(time.clock()- sttime)+ " Secs.\n")
'''
