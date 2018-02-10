import nltk
import nltk.tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import os
import xlrd
import re

data = []

workbook = xlrd.open_workbook('articles.xlsx')

sheet = workbook.sheet_by_name('articles.full')

tagsToRemove = ['PRP','RB','IN','DT','CD','TO','CC']

for row in range(1,sheet.nrows):
    line=sheet.cell(row, 2)
    line = word_tokenize(str(line))
    # tokenizer = RegexpTokenizer(r'\w+')
    #
    # tokens=tokenizer.tokenize(str(sheet.cell(row,3)).strip('number:')+str(line))

    tokens = ' '.join(e for e in line if e.isalnum())
    tokens = str(sheet.cell(row,3)).strip('number:') +' '+ tokens
    tokens = tokens.split(' ')
    #tokens = str(sheet.cell(row,3)).strip('number:')+str(line)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tagged=nltk.pos_tag(tokens)
    taggedFiltered = []
    flg = 0
    for i in range(0,len(tagged)):
        if i == 0:
            taggedFiltered.append(tagged[i])
        elif tagged[i][1] not in tagsToRemove:
            taggedFiltered.append(tagged[i])
    data.append(taggedFiltered)
    print(taggedFiltered)

datatoWrite = []
with open('Preprocessed.csv','w') as f:
    for x in range(0,len(data)):
        write = []
        for y in data[x]:
            if y[0] in ['text','homeless','Homeless','Homelessness','homelessness','HOMELESS','HOMELESSNESS','n']:
                continue
            else:
                write.append(y[0])
                f.write(y[0])
                f.write(' ')
        f.write(',')
        datatoWrite.append(write)
flg = 0
final = []
for x in datatoWrite:
    flg = 0
    lx = ''
    for icounter in range(0,len(x)):
        if flg == 0:
            flg = 1
            continue
        else:
            lx = lx +' '+ x[icounter]
    final.append(lx)
import xlwt

wb = xlwt.Workbook()
ws = wb.add_sheet('Preprocessed')
for i in range(0, len(datatoWrite)):
    line = ''
    for j in range(len(datatoWrite[i])):
        if j == 0:
            ws.write(i,1,datatoWrite[i][j])
            continue
        else:
            line = line + ' ' + datatoWrite[i][j]
    ws.write(i, 0, line)

wb.save('example.xls')


import math
import textblob as tb

for y in range(0,len(final)):
    temp = '"""' + final[y] + '"""'
    final[y] = tb.Sentence(temp)


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


tfIdfed = []
bloblist = final
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    line = ''
    for word, score in sorted_words[:100]:
        line = line + ' ' + word
        print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
    tfIdfed.append(line)
import xlwt

wb = xlwt.Workbook()
ws = wb.add_sheet('Tf-Idf')
for i in range(1, len(tfIdfed)):
    ws.write(i, 1, tfIdfed[i])

wb.save('TF-IdFed.xls')


import xlwt

wb = xlwt.Workbook()
ws = wb.add_sheet('Preprocessed')
for i in range(0, len(datatoWrite)):
    line = ''
    for j in range(len(datatoWrite[i])):
        if j == 0:
            ws.write(i,1,datatoWrite[i][j])
            continue
    ws.write(i, 0, tfIdfed[i])

wb.save('CleanData.xls')

for i in range(len(tfIdfed)):
    tfIdfed[i]= tfIdfed[i].split()


# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(tfIdfed)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in tfIdfed]

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=10, num_words=30))






















