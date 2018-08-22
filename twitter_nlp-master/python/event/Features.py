import sys
import re
import os
import string
import subprocess

#BASE_DIR = '/home/aritter/twitter_nlp'
#BASE_DIR = os.environ['HOME'] + '/twitter_nlp'
#BASE_DIR = '/homes/gws/aritter/twitter_nlp'
BASE_DIR = 'twitter_nlp.jar'

if os.environ.has_key('TWITTER_NLP'):
    BASE_DIR = os.environ['TWITTER_NLP']

#sys.path.append('%s/python/' % (BASE_DIR))
#sys.path.append('%s/python/cap/' % (BASE_DIR))
#sys.path.append('../cap/')
#import cap_classifier

def Brown2Bits(bits):
    bitstring = ""
    for i in range(20):
        if bits & (1 << i):
            bitstring += '1'
        else:
            bitstring += '0'
    return bitstring

def GetOrthographicFeatures(word, goodCap):
    features = []

    #Don't include these features for usernames
    features.append("word=%s" % word)
    features.append("word_lower=%s" % word.lower())
    if(len(word) >= 4):
        features.append("prefix=%s" % word[0:3].lower())
        features.append("prefix=%s" % word[0:4].lower())
        features.append("suffix=%s" % word[len(word)-3:len(word)].lower())
        features.append("suffix=%s" % word[len(word)-4:len(word)].lower())

        #Substring features (don't seem to help)
        #for i in range(1,len(word)-2):
        #    for j in range(i+1,len(word)-1):
        #        features.append("substr=%s" % word[i:j])

#    if re.search(r'^[A-Z]', word):
#        features.append('INITCAP')
#    if re.search(r'^[A-Z]', word) and goodCap:
#        features.append('INITCAP_AND_GOODCAP')
#    if re.match(r'^[A-Z]+$', word):
#        features.append('ALLCAP')
#    if re.match(r'^[A-Z]+$', word) and goodCap:
#        features.append('ALLCAP_AND_GOODCAP')

    #Check for nominalization suffixes (see Bethard et. al.)
    if re.search(r'(ings?|ions?|ments?|nces?)$', word):
        features.append('NOMINALIZATION')

    if re.match(r'.*[0-9].*', word):
        features.append('HASDIGIT')
    if re.match(r'[0-9]', word):
        features.append('SINGLEDIGIT')
    if re.match(r'[0-9][0-9]', word):
        features.append('DOUBLEDIGIT')
    if re.match(r'.*-.*', word):
        features.append('HASDASH')
    if re.match(r'[.,;:?!-+\'"]', word):
        features.append('PUNCTUATION')
    return features

class DictionaryFeatures:
    def __init__(self, dictDir):
        self.brownClusters = None
        self.word2dictionaries = {}
        self.dictionaries = []
        i = 0
        for d in os.listdir(dictDir):
            self.dictionaries.append(d)
            if d == '.svn':
                continue
            for line in open(dictDir + "/" + d):
                word = line.rstrip('\n')
                word = word.strip(' ').lower()
                if not self.word2dictionaries.has_key(word):            #Tab-seperated string is more memory efficient than a list?
                    self.word2dictionaries[word] = str(i)
                else:
                    self.word2dictionaries[word] += "\t%s" % i
            i += 1

    def AddBrownClusters(self, brownFile):
        self.brownClusters = {}
        for line in open(brownFile):
            line = line.rstrip('\n')
            (word, bits) = line.split(' ')
            bits = int(bits)
            self.brownClusters[word] = bits

    MAX_WINDOW_SIZE=6
    def GetDictFeatures(self, words, i):
        features = []
        for window in range(self.MAX_WINDOW_SIZE):
            for start in range(max(i-window+1, 0), i+1):
                end = start + window
                phrase = ' '.join(words[start:end]).lower().strip(string.punctuation)
                #sys.stderr.write("%s\t%s\n" % (words[i], phrase))
                if self.word2dictionaries.has_key(phrase):
                    #features.append('NDICT=%s' % len(self.word2dictionaries[phrase]))
                    for j in self.word2dictionaries[phrase].split('\t'):
                        features.append('DICT=%s' % self.dictionaries[int(j)])
                        #features.append('DICT%s=%s' % (window,d))
                        if window > 1:
                            features.append('DICTWIN=%s' % window)
        if self.brownClusters and self.brownClusters.has_key(words[i].lower()):
            for j in [4, 8, 12]:
                #features.append('BROWN%s=%s' % (str(j), self.brownClusters[words[i].lower()] >> j))
                bitstring = Brown2Bits(self.brownClusters[words[i].lower()])
                features.append('BROWN=%s' % bitstring[0:j+1])                
        return list(set(features))

class DictionaryFeatures2(DictionaryFeatures):
    def __init__(self, dictFile):
        self.word2dictionaries = {}
        for line in open(dictFile):
            (word, dictionary) = line.rstrip('\n').split('\t')
            if re.search(r'^/(common|user|type|freebase|base)/', dictionary):
                #print "%s\t%s" % (word, dictionary)
                continue
            #print dictionary
            if not self.word2dictionaries.has_key(word):
                self.word2dictionaries[word] = []
            self.word2dictionaries[word].append(dictionary)

def GetQuotes(words):
    string = ' '.join(words)
    quoted = []
    string = re.sub(r"' ([^']+) '", r"' |||[ \1 ]||| '", string)
    string = re.sub(r'" ([^"]+) "', r'" |||[ \1 ]||| "', string)

    #sys.stderr.write(string + "\n")

    isquoted = False
    words = string.split(' ')
    for i in range(len(words)):
        if words[i] == "|||[":
            isquoted = True
        elif words[i] == "]|||":
            isquoted = False
        else:
            quoted.append(isquoted)
            
    return quoted

class FeatureExtractor:
    #def __init__(self, dictDir="data/dictionaries", brownFile="/homes/gws/aritter/twitter_nlp/data/brown_clusters/60K_clusters.txt"):
    def __init__(self, dictDir="data/dictionaries", brownFile="%s/data/brown_clusters/60K_clusters.txt" % (BASE_DIR)):
        self.df = DictionaryFeatures(dictDir)
        if brownFile:
            self.df.AddBrownClusters(brownFile)

    LEFT_RIGHT_WINDOW=3
    def Extract(self, words, pos, chunk, i, goodCap=True):
        features = GetOrthographicFeatures(words[i], goodCap) + self.df.GetDictFeatures(words, i) + ["goodCap=%s" % goodCap]

        #sys.stderr.write("%s\t%s\n" % (words, pos))

        for j in range(i-self.LEFT_RIGHT_WINDOW,i+self.LEFT_RIGHT_WINDOW):
            if j > 0 and j < i:
                features.append('LEFT_WINDOW=%s' % words[j])
            elif j < len(words) and j > i:
                features.append('RIGHT_WINDOW=%s' % words[j])

        if pos:
            features.append('POS=%s' % pos[i])
            features.append('POS=%s' % pos[i][0:1])
            features.append('POS=%s' % pos[i][0:2])

        if chunk:
            features.append('CHUNK=%s' % chunk[i])

        if i == 0:
            features.append('BEGIN')

        if pos:
            features.append('POS=%s_X_%s' % ('_'.join(pos[i-1:i]),'_'.join(pos[i+1:i+2])))
        if chunk:
            features.append('CHUNK=%s_X_%s' % ('_'.join(chunk[i-1:i]),'_'.join(chunk[i+1:i+2])))

        if i > 0:
            features += ["p1=%s" % x for x in  GetOrthographicFeatures(words[i-1], goodCap) + self.df.GetDictFeatures(words, i-1)]
            if pos:
                features.append('PREV_POS=%s' % pos[i-1])
                features.append('PREV_POS=%s' % pos[i-1][0:1])
                features.append('PREV_POS=%s' % pos[i-1][0:2])
#            if chunk:
#                features.append('PREV_CHUNK=%s' % chunk[i-1])
        if i > 1:
            if pos:
                features.append('PREV_POS=%s_%s' % (pos[i-1], pos[i-2]))
                features.append('PREV_POS=%s_%s' % (pos[i-1][0:1], pos[i-2][0:1]))
                features.append('PREV_POS=%s_%s' % (pos[i-1][0:2], pos[i-2][0:2]))
#            if chunk:
#                features.append('PREV_CHUNK=%s_%s' % (chunk[i-1], chunk[i-2]))
        if i < len(words)-1:
            features += ["n1=%s" % x for x in  GetOrthographicFeatures(words[i+1], goodCap) + self.df.GetDictFeatures(words, i+1)]
            if pos:
                features.append('NEXT_POS=%s' % pos[i+1])
                features.append('NEXT_POS=%s' % pos[i+1][0:1])
                features.append('NEXT_POS=%s' % pos[i+1][0:2])
#            if chunk:
#                features.append('NEXT_CHUNK=%s' % chunk[i+1])
        if i < len(words)-2:
            if pos:
                features.append('NEXT_POS=%s_%s' % (pos[i+1], pos[i+2]))
                features.append('NEXT_POS=%s_%s' % (pos[i+1][0:1], pos[i+2][0:1]))
                features.append('NEXT_POS=%s_%s' % (pos[i+1][0:2], pos[i+2][0:2]))
#            if chunk:
#                features.append('NEXT_CHUNK=%s_%s' % (chunk[i+1], chunk[i+2]))
        return features
