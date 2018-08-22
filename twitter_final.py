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
##########################


from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D,GRU,Activation
from keras.layers.core import Flatten, Dense, Dropout,Reshape
from keras.models import Sequential
from keras import backend as K

######################################
'''
#one hot representation
Here we make our one hot encoder for numbers ,alphabets,special characters 
'''

numbers="0123456789"
alphabetes="abcdefghijklmnopqrstuvwxyz"
special_char= " $%&'()*+,-./:;<=>?@[\]^_`{|}~!"
special_char=special_char+'"'
characters=numbers+alphabetes+special_char

no_characters=len(numbers)+len(alphabetes)+len(special_char)
max_len=160
#^^tweets max length

'''
#a dictionary for indicating different characters with a increasing counter
#used when we have to make the one hot representation for each character in a tweet
'''
char_dict=dict()
for i in range(len(characters)):
	char_dict.update( { characters[i]:i } )

'''
#opeing of files the list of entity and hashtags
the list formed named list_list contains all entity and hashtags
'''
list_read= open("twitter_final /files/passing_files/final_entity_hashtag_list.txt", "r")
list_list = list_read.read().split("\n")
list_list = list_list[:-1]

'''
#creating a dictionary with entity as keys
It will be used when we want to make the label targets for the CNN model
'''
entity_dict =dict()
counter=0
for x in list_list:
	entity_dict.update({x:counter})
	counter=counter+1

#dictionary of final labels
label_dict=dict()
label_dict.update({"false":0})
label_dict.update({"true":1})
label_dict.update({"unverified":2})
label_dict.update({"non-rumor":3})

######################################################################
#opening file of ids with their tweets
with codecs.open("twitter_final /files/passing_files/tweets.txt", "r",encoding='utf-8', errors='ignore') as fdata:
	tweet_lines = fdata.read().split("\n")

#division of first data in two columns
ID = ["" for x in range(len(tweet_lines))]
TWEET = ["" for x in range(len(tweet_lines))]
tweet_dict=dict()

'''
#preprocessing of first data

#Here we take the data of tweets and their ids and 
then create a new file consisting of only tweets for the further use
and also created a dictionary ( tweet_dict )with the tweet ids as the dictionary keys
and the tweets as their respective values
'''

ii=0
for l in tweet_lines:
	sr= l
	if not sr.strip():
        	continue
	else:
		sr = sr.split("\t")
		if len(sr)==2:
			ID[ii]=sr[0]
			TWEET[ii]=sr[1]
			ii=ii+1
		else :
			TWEET[ii-1]=TWEET[ii-1]+' '+l
file1= open("set of tweets", "w")

'''
#in the following code the hyperlink present in the tweet is removed
and the dictionary is formed
 '''
for i in range(ii):
	url=r'[a-z]*[:.]+\S+'
	#following statement removes the hyperlink
	TWEET[i]=re.sub(url,'',TWEET[i].lower())
	tweet_dict.update({ID[i]:TWEET[i]})
	file1.write(TWEET[i])
	file1.write('\n')
file1.close()

#accesing the file having only tweets
tweet_read= open("twitter_final /files/passing_files/only_tweets.txt", "r")
tweets= tweet_read.read().split("\n")
tweets= tweets[:-1]


'''
#this function returns the output of the required layer
inputs- variable name of the model , layer number of the required layer, a variable of the input variable
output-  activation values of the required layer
'''
def get_activations(model, layer, X_batch):
    # get_activations = theano.function([model.layers[0].input, theano.learning_phase()], model.layers[layer].output)
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
    activations = get_activations([X_batch,0])
    return activations
########################################################################
'''
# A funtion is defined to return
  the required matrix for the tweet and its respective label

 Parameters- set of tweets variable, dictionary of the entities, batch_size value, dict of charcters
 Output- on every call it yields the required input of the CNN Model 

'''
def generator(tweets,entity_dict,batch_size,char_dict):
	#for 1 d
	batch_features = np.zeros((batch_size, no_characters, max_len))
	batch_labels = np.zeros((batch_size,len(list_list)))
	index=0
	while True:
		#repaeating on every batch_size
		for i in range(batch_size):
			#iterating over the tweets list
			tweet=tweets[index]
			tweet=tweet.lower()
			#1d
		    #this matrix store matrix one hot representation of ev0ery tweet
			tweet_matrix=np.zeros((no_characters,max_len))

			k=0
			#iterating over one tweeet
			#accessing every letter
			for j in range(len(tweet)):
				ch=tweet[j]
				#Forming a matriix representation of the tweet
				if characters.find(ch)!=-1:
					tweet_matrix[char_dict[ch],k]=1
					k=k+1
			#this matrix store matrix one hot representation of lable of every tweet
			label_matrix=np.zeros((len(list_list)))
			#assigning values in matrices
			for key,value in entity_dict.items():
				if tweet.find(key)!=-1:
					label_matrix[entity_dict[key]]=1

			#equating the tweet mtrix formed and the input 
			batch_features[i]= tweet_matrix
			batch_labels[i] = label_matrix
			index=index+1
			if(index==len(tweets)):
				index=0
		#yield of the required things
		yield batch_features, batch_labels

#opening all ids with their labels
train_data_open= open("twitter_final /train_and_test/twitter15_train_shuf.txt", "r")
train_id_data = train_data_open.read().split("\n")
train_id_data=train_id_data[:-1]

test_data_open= open("twitter_final /train_and_test/twitter15_test_shuf.txt", "r")
test_id_data = test_data_open.read().split("\n")
test_id_data=test_id_data[:-1]


'''
# A funtion is defined to return
  the required activation values for the tweets and its respective  final class label

  Parameters- mean same as the variables written below
  Output- Input and Output for a batch
'''


def generator_gru(train_id_data,tweet_dict,char_dict,label_dict,model,batch_size):
	max_no_id_unique_tweets=356
	final_input=np.zeros((batch_size,max_no_id_unique_tweets, 1024))
	final_output=np.zeros((batch_size,4))
	index=0
	while True:
		for a in range(batch_size):
			#split of the lines in  data 
			id_value=train_id_data[index].split(":")	
			#first value=class name
			#second name= file name/tweet id 		
			tree_id=id_value[1]+".txt"
			read_tree= open("twitter_final /data/twitter15/tree/"+tree_id, "r")
			edge = read_tree.read().split("\n")
			edge=edge[:-1]

			#formation of dict
				#contains the unique ids w.r.t a particular Tweet id
				#Key= Tweet id
				#Value=time gap
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

			#organising the data in a good manner

			#final char_dict
			unique1 =["" for x in range(len(unique))]
			i=0

			#sorting according to time
			for key ,value in sorted(unique.items(), key=itemgetter(1)):
				unique1[i]=key
				i+=1

			#list of tweets of the unique tweet ids
			id_unique_tweets= ["" for x in range(len(unique1))]
			
			for i  in range(len(unique1)):
				if unique1[i] in tweet_dict.keys():
					id_unique_tweets[i]=tweet_dict[unique1[i]]

			final_batch_features = np.zeros((max_no_id_unique_tweets, 1024))
			intermediate_batch_features = np.zeros((max_no_id_unique_tweets,no_characters,max_len))
			
			#formation of the matrices the tweets in unique tweet ids
			i=0
			for tweet in range(len(id_unique_tweets)):
				test_case=id_unique_tweets[tweet]
				test_case=test_case.lower()

				#this matrix store matrix one hot representation of every tweet
				tweet_matrix=np.zeros((no_characters,max_len))
				k=0
				for j in range(len(test_case)):
					ch=test_case[j]
					if characters.find(ch)!=-1:
						tweet_matrix[char_dict[ch],k]=1
						k=k+1
				intermediate_batch_features[i]= tweet_matrix
				i=i+1

			label_matrix=np.zeros((4))
			label_matrix[label_dict[id_value[0]]]=1
			
			#Activation layers for the required input batch 
			x=get_activations(model,9,intermediate_batch_features)

			#Output of above values of x in a nice manner
			for i in range(len(x[0])):
				final_batch_features[i]=x[0][i]
			
			final_input[a]=final_batch_features
			final_output[a]=label_matrix	

			index+=1
			if(index==len(train_id_data)):
				index=0		
		yield final_input, final_output

'''
Defining the matrices parameteres
'''
#######################################################################
def mcor(y_true, y_pred):
     #matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
 
 
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
 
 
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
 
 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
 
 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
 
     return numerator / (denominator + K.epsilon())




def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

#######################################################################

#######################################################################			
#batch_size to train
batch_size = 64
# number of output classes
n_classes = len(list_list)
# number of epochs to train
nb_epoch = 150
# number of convolutional filters to use
nb_filters = 256
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 4

model = Sequential()
model.add(Conv1D(nb_filters, nb_conv, activation='relu', input_shape=(no_characters,max_len)))
model.add(Conv1D(nb_filters, nb_conv,  activation='relu'))

model.add(MaxPooling1D(2))

model.add(Conv1D(nb_filters, nb_conv, activation='relu'))
model.add(MaxPooling1D(2))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(n_classes, activation='softmax'))
#compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',precision,recall])

#fitting the model
model.fit_generator(generator(tweets,entity_dict,batch_size,char_dict), steps_per_epoch = 50,nb_epoch=nb_epoch)

#GRU Model starts

gru_nb_epoch=100
max_no_id_unique_tweets=356
batch_size1=16
#gru model
model1 = Sequential()
model1.add(GRU(64,input_shape=(max_no_id_unique_tweets,1024),return_sequences=True))
model1.add(Flatten())
model1.add(Dense(4, activation='softmax'))
# model1.add(GRU(4, return_sequences=True))
# model1.add(Activation('softmax'))

#compiling
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',precision,recall])
#fiiting the model
model1.fit_generator(generator_gru(train_id_data,tweet_dict,char_dict,label_dict,model,batch_size1),steps_per_epoch = 2,nb_epoch=gru_nb_epoch)
#evaluatating
scores_gru=model1.evaluate_generator(generator_gru(test_id_data,tweet_dict,char_dict,label_dict,model,batch_size1), steps=5)
print("=========")
print (scores_gru)
print("=========")

'''
 #predicting
values=model1.predict_generator(generator_gru(test_id_data,tweet_dict,char_dict,label_dict,model,1), steps=5, max_queue_size=10,  use_multiprocessing=False, verbose=0)
print("=========")
print (values)
print("=========")
'''