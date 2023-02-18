import spacy
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import requests
import sys
from unicodedata import category
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from datetime import date, datetime
import numpy as np
import os

feed_urls = [
    "http://www.lemonde.fr/rss/une.xml",
    "https://www.bfmtv.com/rss/news-24-7/",
    "http://www.lefigaro.fr/rss/figaro_actualites.xml",
    "https://www.lexpress.fr/arc/outboundfeeds/rss/alaune.xml",
    "https://www.francetvinfo.fr/titres.rss",
    "https://www.la-croix.com/RSS",
    "http://tempsreel.nouvelobs.com/rss.xml",
    "http://www.lepoint.fr/rss.xml",
    "https://feeds.leparisien.fr/leparisien/rss",
    "https://www.europe1.fr/rss.xml",
    "https://partner-feeds.20min.ch/rss/20minutes",
    "https://www.afp.com/fr/actus/afp_actualite/792,31,9,7,33/feed"
]


def scrap(feed_urls):
    news_list = pd.DataFrame(columns=('title', 'summary', 'img_url', 'link'))

    for feed_url in feed_urls:
        res = requests.get(feed_url)
        if (res.status_code == 200):
            feed = BeautifulSoup(res.content, features='xml')
            
            articles = feed.findAll('item')
            for article in articles:
                news = {
                    'title': None,
                    'summary': None,
                    'link': None,
                    'img_url': None
                }
                news['title'] = BeautifulSoup(article.find('title').get_text(), "html").get_text()
                if (article.find('description')):
                    news['summary'] = BeautifulSoup(article.find('description').get_text(), "html").get_text()
                if (article.find('content')):
                    news['img_url'] = article.find('content')['url']
                if (article.find('link')):
                    news['link'] = article.find('link').get_text()
                news_list = pd.concat([news_list, pd.DataFrame([news])], ignore_index=True)
        
    return news_list

def process_text(docs, lang='fr'):
    if (lang=='fr'):
        nlp = spacy.load('fr_core_news_lg')
    elif (lang=='en'):
        nlp = spacy.load('en_core_web_sm')

    # Utility functions
    punctuation_chars =  [
        chr(i) for i in range(sys.maxunicode)
        if category(chr(i)).startswith("P")
    ]
    
    lemma_docs = []
    for doc in docs:
        # Tokenize docs
        tokenized_doc = nlp(doc)

        # Lemmanize docs
        lemma_doc = list(filter(lambda token: token.is_stop == False and token.pos_ in ['NOUN', 'PROPN'] and token.lemma_ not in [*string.punctuation, *punctuation_chars], tokenized_doc))
        lemma_doc = list(map(lambda tok: tok.lemma_.lower(), lemma_doc))
        lemma_docs.append(lemma_doc)


    def get_vocabulary_frequency(documents):
        vocabulary = dict()
        for doc in documents:
            for word in doc:
                if word in list(vocabulary.keys()):
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1

        return vocabulary

    voc = get_vocabulary_frequency(lemma_docs)

    return lemma_docs, voc

def output_file(data, filename):
    path = f'./data/{date.today().strftime("%d-%m-%Y")}'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/{filename}', 'w', encoding='UTF8', newline='') as f:
        writer = json.dump(data, f, ensure_ascii=False)

def graphnet(docs, voc, min_freq=5):
    
    # Filter voc with min_freq
    filtered_voc = dict(filter(lambda elem: elem[1] > min_freq, voc.items()))

    dict_voc_id = dict()
    for i, term in enumerate(filtered_voc):
        dict_voc_id[term] = i
    
    # List bigrams (edges)
    finder = nltk.BigramCollocationFinder.from_documents(docs)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigrams = list(finder.score_ngrams(bigram_measures.raw_freq))
    min_freq = min(list(map(lambda x: x[1], bigrams)))
    bigrams = list(map(lambda x: (x[0], x[1]/min_freq), bigrams))

    # Filter the bigrams with filtered_voc elements and replace by id
    filtered_bigrams = []
    for bigram in bigrams:
        if (bigram[0][0] in filtered_voc.keys() and bigram[0][1] in filtered_voc.keys()):
            #new_bigram = ( dict_voc_id[bigram[0][0]] , dict_voc_id[bigram[0][1]] )
            new_bigram = bigram[0]
            filtered_bigrams.append((new_bigram, bigram[1]))

    # Set nodes sizes
    sizes = list(filtered_voc.values())

    # Format data
    nodes = []
    for i, term in enumerate(filtered_voc.keys()):
        nodes.append({
            'id': term,
            'label': term,
            'size': sizes[i]
        })
    
    edges = []
    for i, edge in enumerate(filtered_bigrams):
        (source, target) = edge[0]
        edges.append({
            'id': i,
            'source': source,
            'target': target,
            'size': edge[1]
        })

    
    # Write JSON files
    output_file(nodes, 'nodes.json')

    output_file(edges, 'edges.json')

def list_dates():
    dates = [x for x in next(os.walk('./data'))[1]]
    dates.sort(key=lambda date: datetime.strptime(date, "%d-%m-%Y"), reverse=True)
    dates = [{"name": x} for x in dates]
    with open(f'./data/list.json', 'w', encoding='UTF8', newline='') as f:
        writer = json.dump(dates, f, ensure_ascii=False)


def find_topics(docs, vocab_freq, wordcomat, voc2id, k=5):
    topics = []
    for i in range(100):
        term = vocab_freq[i]
        best_terms = find_similar_tops(term, vocab_freq, wordcomat, voc2id, k)
        sims = np.zeros(len(topics))
        if (len(best_terms) == 0):
            continue
        for j, topic in enumerate(topics):
            sims[j] = bow_similarity([term] + [x[1] for x in best_terms], topic, vocab_freq, wordcomat,voc2id, k)
        
        raw_sim = bow_similarity([term], [x[1] for x in best_terms], vocab_freq, wordcomat,voc2id, k)
        if (len(sims) > 0 and np.max(sims) > 0 and np.max(sims) > raw_sim):
                best_topic_id = np.argmax(sims)
                if (term not in topics[best_topic_id]):
                    topics[best_topic_id].append(term)
        else:
            topics.append([term, *[x[1] for x in best_terms]])

    return topics

def top_k_words(word, vocab_freq, wordcomat, voc2id, k):
    word_id = voc2id[word]
    top_k_ids = np.argsort(wordcomat[word_id,:])[::-1][:k]
    return [(wordcomat[word_id, i], vocab_freq[i]) for i in top_k_ids]

def find_similar_tops(word, vocab_freq, wordcomat, voc2id, k):
    top_k = top_k_words(word, vocab_freq, wordcomat, voc2id, k)
    results = []
    for term in top_k:
        if (word in list(map(lambda x: x[1], top_k_words(term[1], vocab_freq, wordcomat, voc2id, k)))):
            results.append(term)
    return results

def similarity(x, y):
    count = 0
    if (len(x) == 0 or len(y) == 0):
        return 0
    for a in x:
        for b in y:
            if (b == a):
                count += 2

    return count/(len(x)+len(y))

def bow_similarity(b1, b2, vocab_freq, wordcomat, voc2id, k):
    sim1 = []
    for x in b1:
        for w in find_similar_tops(x, vocab_freq, wordcomat, voc2id, k):
            sim1.append(w[1])
    sim2 = []
    for x in b2:
        for w in find_similar_tops(x, vocab_freq, wordcomat, voc2id, k):
            sim2.append(w[1])
    return similarity(sim1, sim2)

def construct_word_cooccurence_matrix(voc2id, documents):
    matrix = np.zeros((len(voc2id), len(voc2id)))
    for document in documents:
        if (len(document) > 1):
            for word_i in document:
                for word_j in document:
                    if (word_i != word_j):
                        matrix[voc2id[word_i], voc2id[word_j]] += 1
    

    return matrix/matrix.sum(axis=1, keepdims=True)



def create_vocabulary_frequency(corpus):
    '''Select top-k (k = vocab_len) words in term of frequencies as vocabulary'''
    voc2id = {}
    count = dict()
    for document in corpus:
        if (len(document)>1):
            for word in document:
                word = word.lower()
                if (word in count):
                    count[word] += 1
                else:
                    count[word] = 1
            
    
    sorted_count_by_freq = sorted(count.items(), key=lambda kv: kv[1], reverse=True)
    
    vocabulary = []
    for i, x in enumerate(sorted_count_by_freq):
        vocabulary.append(x[0])
        voc2id[x[0]] = i
    return vocabulary, voc2id




def find_similarities(trend, docs, threshold=0.3):
    results = []
    for i, doc in enumerate(docs):
        sim = similarity(trend, doc)
        if (sim > threshold):
            results.append((i, sim, news_list.iloc[i]['link']))
    results = sorted(results, key=lambda x: -x[1])
    return results

def find_trends(topics, docs, threshold=0.3):
    trends = []
    for topic in topics:
        similar_docs = find_similarities(topic, docs, threshold)
        img = None
        if (len(similar_docs) == 0):
            continue
        for doc in similar_docs:
            if (news_list.iloc[doc[0]]['img_url']):
                img = news_list.iloc[doc[0]]['img_url']
                break
        trends.append({
            "topic": topic,
            "docs": similar_docs,
            "title": news_list.iloc[similar_docs[0][0]]['title'] ,
            "img_url": img
        })
    
    return trends


news_list = scrap(feed_urls)
output_file(list(map(lambda x: {**x[1], "news_id": x[0]}, news_list.T.to_dict().items())), 'news.json')

docs, voc = process_text(news_list['title'], lang='fr')

graphnet(docs, voc, min_freq=2)

vocab_freq, voc2id = create_vocabulary_frequency(docs)
wordcomat = construct_word_cooccurence_matrix(voc2id, docs)

topics = find_topics(docs, vocab_freq, wordcomat,voc2id, k=10)

trends = find_trends(topics, docs, 0.5)
output_file(trends, 'trends.json')
list_dates()
