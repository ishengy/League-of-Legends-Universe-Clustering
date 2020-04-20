# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:57:27 2019

@author: ivan.sheng
"""
from __future__ import division
from bs4 import BeautifulSoup 
import re
import pandas as pd 
import os
import time

from selenium import webdriver

import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from gensim import corpora, models
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

path = 'C:/Users/ivan.sheng/Downloads/'
os.chdir(path)

file_name = 'C:/Users/ivan.sheng/Downloads/champList.xlsx' # path to file + file name
sheet = 'champs' # sheet name or sheet number or list of sheet numbers and names
champList = pd.read_excel(io=file_name, sheet_name=sheet)

sheet = 'locations' # sheet name or sheet number or list of sheet numbers and names
locList = pd.read_excel(io=file_name, sheet_name=sheet)

driver = webdriver.Firefox(executable_path = r'C:\Users\ivan.sheng\Downloads\geckodriver-v0.24.0-win64\geckodriver.exe')
driver.implicitly_wait(30)

html = []

for champ in champList['Champion']:
    base_url = "https://universe.leagueoflegends.com/en_US/story/champion/" + champ
    verificationErrors = []
    accept_next_alert = True
    driver.get(base_url)
    time.sleep(1.2)
    html_source = driver.page_source
    html.append(html_source)

#for loc in locList['Region']:
#    base_url = "https://universe.leagueoflegends.com/en_US/region/" + loc
#    verificationErrors = []
#    accept_next_alert = True
#    driver.get(base_url)
#    time.sleep(1.2)
#    html_source = driver.page_source
#    html.append(html_source)

results = []
for body in html:
    data = body.encode('utf-8')
    bs = BeautifulSoup(data, 'html.parser')
    text = ''
    
    for content in bs.find_all('p', class_='p_1_sJ'):
        text += content.text + ' '
    if len(text) < 4:
        for content in bs.find_all('div', class_='root_3nvd dark_1RHo'):
            text += content.text + ' '
#        for content in bs.find_all('p', class_='description_1-6k'):
#            text += content.text + ' '
        
    item = {'post': text} # make a dictionary called item, with "post" as the key and the post text as the values
    results.append(item) # append the dictionary to the results list
        
champ_results = pd.DataFrame(results)
out = pd.ExcelWriter('champ_results.xlsx')
champ_results.to_excel(out,'Data')
out.save()

## if already scrapped, just uncomment and run below:
#champ_results = pd.read_excel(io='C:/Users/ivan.sheng/Downloads/champ_results.xlsx', sheet_name='Data')
#champ_results.drop('Unnamed: 0',axis=1,inplace = True)
##

##nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 
lemma = []
for story in champ_results.post:
    word_list = nltk.word_tokenize(story)
    filtered_sentence = [w for w in word_list if not w in stop_words] 
    lemma_out = ' '.join([lemmatizer.lemmatize(w) for w in filtered_sentence])
    lemma.append(lemma_out)

champ_results['Champion'] = champList.Champion
champ_results['Lemmatized'] = lemma

pickle.dump( champ_results, open( "champ_results.p", "wb" ))

# Toeknize, Romove puctuation, Words less than 3 letters
def tokenize(text):
    return re.findall("[a-z']{3,}",text.lower()) 

formatted = champ_results['Lemmatized'].apply(lambda x: tokenize(x))

# BoW + TF-IDF
dictionary = corpora.Dictionary(formatted)
corpus = [dictionary.doc2bow(tokens) for tokens in formatted] # Count words in each row, each word has a unique index
tfidf = models.TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]

# LDA
# Determine best number of clusters
def compute_coherence_values(dictionary, tfidf_corpus, corpus, start, stop, step):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for number_of_topics in range(start, stop, step):
        # generate LDA model
        lda = models.LdaModel(tfidf_corpus, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(lda)
        coherencemodel = models.CoherenceModel(model=lda, texts=formatted, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return coherence_values

## test
lda = models.LdaModel(tfidf_corpus, num_topics=7, id2word = dictionary)  # train model
coherencemodel = models.CoherenceModel(model=lda, texts=formatted, dictionary=dictionary, coherence='c_v')
coherencemodel.get_coherence()
##

# Plot coherence chart
cv = compute_coherence_values(dictionary, tfidf_corpus, corpus, 2, 11, 1)
x = range(2, 11, 1)
plt.plot(x, cv)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Sample lda model
num_topics = 13
lda = models.LdaModel(tfidf_corpus, id2word=dictionary, num_topics=num_topics, random_state = 42)
lda.show_topics(num_topics,num_words=25,formatted=False)
print('\nPerplexity:', lda.log_perplexity(tfidf_corpus))

# Final results - Find the topic number with highest percentage contribution for each article
scores = []
for i in tfidf_corpus:
    scores.append(lda[i])

champ_results['cluster_scores'] = scores

topics = []
for i, row in enumerate(champ_results['cluster_scores']):
    row = sorted(row, key=lambda x: (x[1]), reverse=True)
    for j, (topic_num, prop_topic) in enumerate(row):
        if j == 0:
            topics.append(int(topic_num))

main_idea = lda.print_topics(num_words=10)
for t in main_idea:
    print(t)

# Sparse + Cosine Similarity
def get_cosine_sim_sparse(strs):
    vec = CountVectorizer(token_pattern = "[a-z]{3,}")
    sparse_matrix = vec.fit_transform(strs)
    df = pd.DataFrame(sparse_matrix.toarray(), columns = vec.get_feature_names())
    return (cosine_similarity(df, df))

cosine_sparse = get_cosine_sim_sparse(lemma)

champ_results['final_cluster'] = topics
champ_results.groupby('final_cluster').count() # How many articles in each cluster

out = pd.ExcelWriter('champ_clusters.xlsx')
champ_results.to_excel(out,'clusters')
out.save()
