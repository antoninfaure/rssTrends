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
        lemma_doc = list(map(lambda tok: tok.lemma_, lemma_doc))
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


def find_topics(docs, criterion='leverage', level=0.01):
    te = TransactionEncoder()
    te_ary = te.fit(docs).transform(docs, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True, verbose=1)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    rules = association_rules(frequent_itemsets, metric ="lift", min_threshold = 1)
    rules = rules.sort_values([criterion], ascending =[False])

    rules = rules[rules[criterion] > level]

    topics = []
    for i in rules.index:
        rule = rules.loc[i]
        x = list(rule['antecedents'])
        y = list(rule['consequents'])
        terms = x + y
        found_similar = False
        delete_topics_ids = []
        for i, topic in enumerate(topics):
            sim = similarity(topic, terms)
            if (similarity(topic, terms) > 0.2):
                found_similar = True
                new_topic = list(set(list(topic) + terms))
                delete_topics_ids.append(i)
                break
        if (found_similar == False):
            topics.append((tuple(terms)))
        else:
            topics = [x for i, x in enumerate(topics) if i not in delete_topics_ids]
            topics.insert(min(delete_topics_ids), tuple(new_topic))

    return topics

def list_dates():
    dates = [x for x in next(os.walk('./data'))[1]]
    dates.sort(key=lambda date: datetime.strptime(date, "%d-%m-%Y"), reverse=True)
    dates = [{"name": x} for x in dates]
    with open(f'./data/list.json', 'w', encoding='UTF8', newline='') as f:
        writer = json.dump(dates, f, ensure_ascii=False)

def similarity(x, y):
    count = 0
    for a in x:
        for b in y:
            if (b == a):
                count += 1
    return count/len(x)

def find_similarities(trend, docs, threshold=0.3):
    results = []
    for i, doc in enumerate(docs):
        sim = similarity(trend, doc)
        if (sim > threshold):
            results.append((i, sim, news_list.iloc[i]['link']))
    results = sorted(results, key=lambda x: -x[1])
    return results

def find_trends(topics, docs):
    trends = []
    for topic in topics:
        similar_docs = find_similarities(topic, docs)
        img = None
        for doc in similar_docs:
            if (news_list.iloc[doc[0]]['img_url']):
                img = news_list.iloc[doc[0]]['img_url']
                break
        trends.append({
            "topic": topic,
            "docs": similar_docs,
            "img_url": img
        })
    
    return trends


news_list = scrap(feed_urls)
output_file(list(map(lambda x: {**x[1], "news_id": x[0]}, news_list.T.to_dict().items())), 'news.json')

docs, voc = process_text(news_list['title'], lang='fr')

graphnet(docs, voc, min_freq=2)

topics = find_topics(docs, 'leverage', 0.005)

trends = find_trends(topics, docs)
output_file(trends, 'trends.json')
list_dates()
