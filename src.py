from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import random
import math
import re
import time
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

lancaster = LancasterStemmer()
retoken = RegexpTokenizer(r"[\w]+")                         #Tokenize and remove punctuation marks. 
sw = stopwords.words()

bag = {}
bag_freq = {}
knn = 81
sentence_vector_list_train = []                             
sentence_vector_list_test = []
kfoldaccuracies = []
f = open("prediction.txt", "a")                                 #Opening file to write predicted labels.

def reader(typeofdata):
    print('read')
    with open(typeofdata,'r',encoding='utf8') as f:
        f_contents = f.read()                          #Read contents of file
        f_contents_list = f_contents.split("#EOF")     #split sentences in file with delimiter as '#EOF'
        f_contents_list.pop()                          #remove last empty list
    return f_contents_list                             #List of lists of reviews
 
def preprocessing(f_contents_list,typeofdata):
    print('preprocess')
    counter = 1
    label=[]
    sentence_vector_list = []
    preprocess=1
    for str1 in f_contents_list:
        sentence_vector = {}
        print(preprocess)
        preprocess+=1
        str1=str1.replace("+1","positive")
        str1=str1.replace("-1","negative")
        text_tokens = retoken.tokenize(str1.lower())          #pass string with all lowercase characters
        if (typeofdata=='train.txt' or typeofdata=='kfoldtest.txt'): #below 2 operations only if using training set
            if text_tokens[0]=='positive':                      #create class list of training set
                label.append(1)                                 
            elif text_tokens[0]=='negative': 
                label.append(-1)
            text_tokens.pop(0)                                  #remove class labels from training reviews
            
        for word in text_tokens:
            if word not in sw:
                stemmedword = lancaster.stem(word)              #stemming
                if (stemmedword not in bag and typeofdata=='train.txt'):    #Counter variable used below is index of bag of words
                    bag[stemmedword] = counter                  #adding words from training data to bag of words
                    bag_freq[stemmedword] = 1                   #maintaining frequency of words in bag
                    sentence_vector[counter] = 1                #Convert training review into dictionary which stores index of word in bag of words as key and the word's frequency in the particular training review as value of dicitonary.
                    counter+=1
                elif(stemmedword in bag):
                    bag_freq[stemmedword] += 1                  #increase frequency of word in bag of words if it already exists in the bag of words
                    if (bag[stemmedword] not in sentence_vector):
                        sentence_vector[bag[stemmedword]] = 1   #increase frequency of word in dictionary representation of that particular review
                    else: 
                        sentence_vector[bag[stemmedword]] += 1  #increase frequency of word in dictionary representation of that particular review
        sentence_vector_list.append(sentence_vector)
    return sentence_vector_list,label

def bag_freq_reduction():   #Function to remove words with low frequency accross entire train review set
    for i in bag_freq:
        if bag_freq[i]<5:
            bag.pop(i)

def similarity_cs(testreview,trainreview,cosine_similarity,trainReviewNumber):
    numer = 0
    dena = 0        
    #Implementing cosine similarity below by comparing each dictionary representation of test reviews with 25000 dictionary representations of training reviews
    for key1,val1 in testreview.items():
        if key1 in trainreview:
            numer += val1*trainreview[key1] #numerator of cosine similarity formula
        dena += val1*val1
    denb = 0
    for val2 in trainreview.values():
        denb += val2*val2
    if len(testreview)==0:
        cosine_similarity[trainReviewNumber] = 0
    elif len(testreview)>0:
        cosine_similarity[trainReviewNumber] = numer/math.sqrt(dena*denb) 
    #trainReviewNumber += 1
    return cosine_similarity    #Returning dictionary with key = training review number and value = cosine similarity between the current test and train review

#Manhattan distance
def similarity_manhattan(testreview,trainreview,manhattan_dist,trainReviewNumber):  
    dist = 0 
    for key1,val1 in testreview.items():
        if key1 in trainreview:
            dist += abs(val1-trainreview[key1])
        else:
            dist += val1
    for key2 in trainreview.keys():
        if key2 not in testreview:
            dist += trainreview[key2]
    manhattan_dist[trainReviewNumber] = dist
    return manhattan_dist

def review_predictor(sorted_similarity,label):
    testlabel = 0
    sorted_similarity_keys = sorted_similarity.keys()   # getting keys representing index(number) of training review from the k nearest neighbor dictionary obtained earlier
    for i in sorted_similarity_keys:
        testlabel+=label[i]             #Summing the labels of k nearest train reviews  
    #if the above sum returns value less than 0 then the predicted test label is -1 and +1 otherwise
    f = open("prediction.txt", "a")
    if testlabel < 0:
        f.write('-1\n')                 #Writing predicted labels to file
        predicted_label = -1
    elif testlabel > 0:
        f.write('+1\n')                 #Writing predicted labels to file
        predicted_label = 1
    f.close()
    return predicted_label


#K fold cross validation with folds = 10
def crossValidation():
    num_folds = 10          #number of folds
    current_fold = 1        
    print('crossvalidation')
    f_contents_list_train = reader('train.txt')
    random.shuffle(f_contents_list_train)               #Shuffling the data
    train_datasize = len(f_contents_list_train)         #Size of complete training data i.e., 25000
    test_datasize = train_datasize/num_folds            #Calculating test data size of cross validator
    while current_fold<=10:
        lowerindex = int(test_datasize*(current_fold-1))    #lowerindex in  of test data of current fold
        higherindex = int(test_datasize*current_fold)       #higherindex of test data of current fold
        newlisttest=f_contents_list_train[lowerindex:higherindex]   #slicing list of training reviews to obtain test data for current fold
        newlisttrain=f_contents_list_train[:lowerindex] + f_contents_list_train[higherindex:] #slicing list of training reviews to obtain train data for current fold
        sentence_vector_list_train,train_label = preprocessing(newlisttrain,'train.txt')    #preprocessing train data, creating bag of words, index dictionaries for each training review
        bag_freq_reduction()                                #Function to remove words with low frequency accross entire train review set
        sentence_vector_list_test,test_label = preprocessing(newlisttest,'kfoldtest.txt')       #preprocessing test data, and creating index dictionaries for each test review
        predicted_label = knearest(sentence_vector_list_train,train_label,sentence_vector_list_test,test_label)     #K nearest neighbors

        predicted_label=[int(i) for i in predicted_label]   #Casting each label in predicted list to int
        test_label=[int(i) for i in test_label]             #Casting each label in test label list to int
  
        kfoldaccuracies.append(metrics.accuracy_score(test_label,predicted_label))  #Accuracy
        
        conf_matrix = confusion_matrix(test_label,predicted_label,labels=[1,-1])    #Confusion matrix
        print('Confusion matrix : \n',conf_matrix)

        conf_matrix_display=ConfusionMatrixDisplay(conf_matrix,display_labels=[1,-1]).plot()
        conf_matrix_display.ax_.set(
                        title='Confusion Matrix for movie review classification', 
                        xlabel='Predicted Labels', 
                        ylabel='Actual Labels')
      
        current_fold += 1
    print('Below is a list of accuracies from cross validation for each fold')
    print(kfoldaccuracies)

def knearest(sentence_vector_list_train,train_label,sentence_vector_list_test,test_label):
    review_counter = 1
    predicted_label = []
    print('cosine_similarity')
    for testreview in sentence_vector_list_test:    #looping over test review
        print('review_counter')
        print(review_counter)
        review_counter+=1
        #sorted_similarity = similarity_cs(testreview,sentence_vector_list_train)
        cosine_similarity = {}
        manhattan_dist = {}
        trainReviewNumber = 0
        
        #Cosine similarity
        for trainreview in sentence_vector_list_train:  #looping over 25000 training reviews for each test review
            cosine_similarity = similarity_cs(testreview,trainreview,cosine_similarity,trainReviewNumber)   #returned dictionary which contains cosine similarity of a single test review with 25000 training reviews
            #print('rnumber')
            #print(trainReviewNumber)
            trainReviewNumber += 1
        sorted_similarity = dict(sorted(cosine_similarity.items(), key=lambda x:x[1], reverse=True)[:knn])    #sorting output in decreasing order and slicing upto the value of k nearest neighbors to be considered

        '''
        #Manhattan distance
        for trainreview in sentence_vector_list_train:
            manhattan_dist = similarity_manhattan(testreview,trainreview,manhattan_dist,trainReviewNumber)
        sorted_similarity = dict(sorted(manhattan_dist.items(), key=lambda x:x[1], reverse=False)[:knn])        
        '''
        predicted_label.append(review_predictor(sorted_similarity,train_label)) #Passing all train review labels and the k nearest reviews generated in above step to predict the label of test review
    return predicted_label

def main():  
    f_contents_list_train = reader('train.txt')
    sentence_vector_list_train,train_label = preprocessing(f_contents_list_train,'train.txt')
    bag_freq_reduction()
    f_contents_list_test = reader('test.txt')
    sentence_vector_list_test,test_label = preprocessing(f_contents_list_test,'test.txt')
    
    knearest(sentence_vector_list_train,train_label,sentence_vector_list_test,test_label)
    
    #crossValidation()  
    
if __name__=="__main__":
    main()