#twitter assignment
#----------------------#
#----------------------#

#libraries
import numpy as np
import numpy.matlib
import codecs
import re
import random
from operator import itemgetter
import json
#import theano
##########################


from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D,GRU,Activation
from keras.layers.core import Flatten, Dense, Dropout,Reshape
from keras.models import Sequential
from keras import backend as K

#opening all ids with their labels
train_data_open= open("/media/code_drunk/5C7B9D870AB33716/backup/twitter_final /train_and_test/twitter15_train_shuf.txt", "r")
train_id_data = train_data_open.read().split("\n")
train_id_data=train_id_data[:-1]

file1=open("test_file","w")
maxa=0
for index in range(len(train_id_data)):
	id_value=train_id_data[index].split(":")			
	tree_id=id_value[1]+".txt"
	read_tree= open("/media/code_drunk/5C7B9D870AB33716/backup/twitter_final /data/twitter15/tree/"+tree_id, "r")
	edge = read_tree.read().split("\n")
	edge=edge[:-1]
	index+=1

	unique =dict()
	#traversing the data 	
	ii=0
	for e in edge:
		li = e
		li=li.split("'")[1::2]
		if ii==0:
			temp=li[4]
		else:
			if li[4]!=temp:
				unique.update( { li[4]:float(li[5]) } )
			if li[1]!=temp:
				unique.update( { li[1]:float(li[2]) } )
		ii=ii+1
	length=str(len(unique))
	maxa=max(maxa,len(unique))
	# print(type(length))
	file1.write(length)
	file1.write('\n')

	unique1 =["" for x in range(len(unique))]
	i=0
	for key ,value in sorted(unique.items(), key=itemgetter(1)):
		unique1[i]=key
		i+=1

	# id_unique_tweets= ["" for x in range(len(unique1))]
	# #print(unique1)
	# for i  in range(len(unique1)):
	# 	if unique1[i] in tweet_dict.keys():
	# 		id_unique_tweets[i]=tweet_dict[unique1[i]]
	# file1.write(str(id_unique_tweets))
	# file1.write('\n')
print(maxa)

