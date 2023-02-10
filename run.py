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

feed_urls = [
    "http://www.lemonde.fr/rss/une.xml",
    "https://www.bfmtv.com/rss/news-24-7/",
    "https://www.liberation.fr/rss/",
    "http://www.lefigaro.fr/rss/figaro_actualites.xml",
    "https://www.franceinter.fr/rss",
    "https://www.lexpress.fr/arc/outboundfeeds/rss/alaune.xml",
    "https://www.francetvinfo.fr/titres.rss",
    "https://www.la-croix.com/RSS",
    "http://tempsreel.nouvelobs.com/rss.xml",
    "http://www.lepoint.fr/rss.xml",
    "https://www.france24.com/fr/rss",
    "https://feeds.leparisien.fr/leparisien/rss",
    "https://www.ouest-france.fr/rss/une",
    "https://www.europe1.fr/rss.xml",
    "https://partner-feeds.20min.ch/rss/20minutes",
    "https://www.afp.com/fr/actus/afp_actualite/792,31,9,7,33/feed"
]

def scrap(feed_urls):
    news_list = pd.DataFrame(columns=('title', 'summary'))

    for feed_url in feed_urls:
        res = requests.get(feed_url)
        feed = BeautifulSoup(res.content, features='xml')

        articles = feed.findAll('item')       
        for article in articles:
            title = BeautifulSoup(article.find('title').get_text(), "html").get_text()
            summary = ""
            if (article.find('description')):
                summary = BeautifulSoup(article.find('description').get_text(), "html").get_text()
            news_list.loc[len(news_list)] = [title, summary]

    return news_list

def process_text(docs, lang='fr'):
    if (lang=='fr'):
        nlp = spacy.load('fr_core_news_sm')
    elif (lang=='en'):
        nlp = spacy.load('en_core_web_sm')

    # Utility functions
    punctuation_chars =  [
        chr(i) for i in range(sys.maxunicode)
        if category(chr(i)).startswith("P")
    ]
    def tokenize(text):
        text = "".join(list(filter(lambda x: x not in [*string.punctuation, *punctuation_chars], text)))
        tokens = nltk.word_tokenize(text)
        words = list(filter(lambda x: x not in [stopwords.words('english') + stopwords.words('french')], tokens))
        return list(map(lambda x: x.lower(), words))

    def preprocess_text(documents):
        docs = list(map(lambda doc: tokenize(doc), documents))
        return docs
    
    # Clean and tokenize docs
    tokenized_docs = preprocess_text(docs)
    
    # Lemmanize docs
    def lemmanize(doc):
        doc = list(filter(lambda token: token.lemma_ not in nlp.Defaults.stop_words, doc))
        return list(map(lambda token: token.lemma_, doc))

    lemma_docs = list(map(lambda doc: lemmanize(nlp(" ".join(doc))), tokenized_docs))
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

def graphnet(docs, voc, min_freq=5, output_url='graph.html'):
    
    # Filter voc with min_freq
    filtered_voc = dict(filter(lambda elem: elem[1] > min_freq, voc.items()))

    dict_voc_id = dict()
    for i, term in enumerate(filtered_voc):
        dict_voc_id[term] = i
    
    # List bigrams (edges)
    finder = nltk.BigramCollocationFinder.from_documents(docs)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigrams = list(finder.score_ngrams(bigram_measures.raw_freq))
    bigrams = list(map(lambda x: x[0], bigrams))

    # Filter the bigrams with filtered_voc elements and replace by id
    bigrams = list(filter(lambda x: x[0] in filtered_voc.keys() and x[1] in filtered_voc.keys(), bigrams))
    bigrams = list(map(lambda x: (dict_voc_id[x[0]], dict_voc_id[x[1]]), bigrams))

    # Set nodes sizes
    sizes = list(filtered_voc.values())

    # Format data
    nodes = []
    for i, term in enumerate(filtered_voc.keys()):
        nodes.append({
            'id': i,
            'label': term,
            'size': sizes[i]
        })
    
    edges = []
    for i, edge in enumerate(bigrams):
        (source, target) = edge
        edges.append({
            'id': i,
            'source': source,
            'target': target
        })

    
    # Write JSON files
    with open('nodes.json', 'w', encoding='UTF8', newline='') as f:
        writer = json.dump(nodes, f, ensure_ascii=False)

    
    with open('edges.json', 'w', encoding='UTF8', newline='') as f:
        writer = json.dump(edges, f, ensure_ascii=False)


news_list = scrap(feed_urls)
docs, voc = process_text(news_list['title'], lang='fr')
graphnet(docs, voc, min_freq=5)