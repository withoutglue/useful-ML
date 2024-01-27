# Let's combine two tasks: finding the most similar pair of lines and the tf-idf representation.

# Write a program that uses the tf-idf vectors to find the most similar pair of lines in a given data set. 
# You can test your solution with the example text below. Note, however, that your code  needs to work on any text.

# This exercise requires a bit more work than average but you should be able to benefit from what you 
# have done in the previous exercises.

text = '''Humpty Dumpty sat on a wall
Humpty Dumpty had a great fall
all the king's horses and all the king's men
couldn't put Humpty together again'''

import math

def distance(row1, row2):

    # the sum of differences between the occurrences
    # of each word in row1 and row2.
    dist = 0
    for i in range(len(row1)):
        if abs(row1[i]-row2[i]) > 0:
            dist += abs(row1[i]-row2[i])

    return dist

def find_nearest_pair(data):
    N = len(data)
    dist = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                dist[i][j] = np.inf
            else:
                dist[i][j] = distance(data[i], data[j])

    print(np.unravel_index(np.argmin(dist), dist.shape))


def main(text):
    # tasks your code should perform:

    # 1. split the text into words, and get a list of unique words that appear in it
    # a short one-liner to separate the text into sentences (with words lower-cased to make words equal 
    # despite casing) can be done with 



    # split the text first into lines and then into lists of words
    docs = [line.lower().split() for line in text.split('\n')]
    
    

    N = len(docs)

    # create the vocabulary: the list of words that appear at least once
    vocabulary = list(set(text.lower().split()))
    #print(vocabulary)


    # 2. go over each unique word and calculate its term frequency, and its document frequency

    df = {}
    tf = {}
    for word in vocabulary:
        # tf: number of occurrences of word w in document divided by document length
        # note: tf[word] will be a list containing the tf of each word for each document
        # for example tf['he'][0] contains the term frequence of the word 'he' in the first
        # document
        tf[word] = [doc.count(word)/len(doc) for doc in docs]

        # df: number of documents containing word w
        df[word] = sum([word in doc for doc in docs])/N
    
   

    # 3. after you have your term frequencies and document frequencies, go over each line in the text and 
    # calculate its TF-IDF representation, which will be a vector


    # create array of TF-IDF representations
    data = []

    # loop through documents to calculate the tf-idf values
    for doc_index, doc in enumerate(docs):
        tfidf = []
        for word in vocabulary:
            # ADD THE CORRECT FORMULA HERE. Remember to use the base 10 logarithm: math.log(x, 10)
            answer = tf[word][doc_index] * math.log(1/df[word], 10)
            tfidf.append(answer) 
    
        data.append(tfidf)

    # 4. after you have calculated the TF-IDF representations for each line in the text, you need to
    # calculate the distances between each line to find which are the closest.

    

    # change the array to np.array
    data = np.array(data)


    find_nearest_pair(data)



main(text)