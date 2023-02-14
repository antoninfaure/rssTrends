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
from datetime import date

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
            title = BeautifulSoup(article.find('title').get_text(), "html.parser").get_text()
            summary = ""
            if (article.find('description')):
                summary = BeautifulSoup(article.find('description').get_text(), "html.parser").get_text()
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
    min_freq = min(list(map(lambda x: x[1], bigrams)))
    bigrams = list(map(lambda x: (x[0], x[1]/min_freq), bigrams))

    # Filter the bigrams with filtered_voc elements and replace by id
    filtered_bigrams = []
    for bigram in bigrams:
        if (bigram[0][0] in filtered_voc.keys() and bigram[0][1] in filtered_voc.keys()):
            new_bigram = ( dict_voc_id[bigram[0][0]] , dict_voc_id[bigram[0][1]] )
            filtered_bigrams.append((new_bigram, bigram[1]))

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


def find_trends(docs, criterion='leverage'):
    te = TransactionEncoder()
    te_ary = te.fit(docs).transform(docs, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True, verbose=1)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    rules = association_rules(frequent_itemsets, metric ="lift", min_threshold = 1)
    rules = rules.sort_values([criterion], ascending =[False])

    rules = rules[rules[criterion] > 0.005]

    trends = []
    for i in rules.index:
        rule = rules.loc[i]
        x = list(rule['antecedents'])
        y = list(rule['consequents'])
        terms = x + y
        ok = True
        new_trend = terms
        delete_trends_ids = []
        for term in terms:
            for i, trend in enumerate(trends):
                if (term in trend):
                    ok = False
                    old_trend = new_trend
                    new_trend = list(set(new_trend + list(trend)))
                    delete_trends_ids.append(i)
        if (ok == True):
            trends.append((tuple(y + x)))
        else:
            trends = [x for i, x in enumerate(trends) if i not in delete_trends_ids]
            trends.append(tuple(new_trend))

    output_file(trends, 'trends.json')
        
    return trends


news_list = scrap(feed_urls)
docs, voc = process_text(news_list['title'], lang='fr')
graphnet(docs, voc, min_freq=5)
trends = find_trends(docs)