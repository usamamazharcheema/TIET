import os
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch

from lib.logger import logger

import numpy
import scipy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')


embedder = SentenceTransformer('bert-base-nli-mean-tokens')
PREDICT_FILE_COLUMNS = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag']
INDEX_NAME = 'vclaim'

#!python3 -m spacy download en_core_web_lg
#!python3 -m spacy download en
import en_core_web_lg
import spacy
sp_lg = en_core_web_lg.load()
def spacy_large_ner(document):
    #document=spacy_large_ner(document)
    #return [(ent.text.strip(), ent.label_) for ent in sp_lg(document).ents]
    return [ent.text.strip().lower() for ent in sp_lg(document).ents]




##### return nouns from POS ###############
nlp = en_core_web_lg.load()

def POS(document):
    doc = nlp(document)
    output=[]
    for token in doc:
        if token.pos_=="NOUN" or token.pos_=="PROPN":
            output.append(token.lemma_.lower())

    return (output)
      
      
      #print(token.text, token.lemma_, token.tag_, token.pos_)
        #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #       token.shape_, token.is_alpha, token.is_stop)


def getCombined(text):
  new=text.replace('#','')
  new=new.replace('@','')
  new=new.replace('.','')
  #new=re.findall('[A-Z][^A-Z]*', new)
  #text=' '.join(new)
  #remove hashtag
  text=new
  pos_tags=POS(text)
  ner_tags= spacy_large_ner(text)
  pos_tags=' '.join(pos_tags)
  pos_tags=pos_tags.split(' ')
  ner_tags= ' '.join(ner_tags)
  ner_tags=ner_tags.split(' ')
  final_named=list(set(pos_tags+ner_tags))
  str1 = ' '.join(final_named)
  return str1

from fuzzywuzzy import fuzz

def get_overlap(vclaim,tweet):
    ff=getCombined(vclaim)
    ss=getCombined(tweet)
    #print(ff)
    #print(ss)
    perc=fuzz.partial_token_sort_ratio(ss, ff)
    #print (perc)
    perc=perc/100
    return perc*80
  #print(fuzz.partial_ratio(s2, s1))


def create_connection(conn_string):
    logger.debug("Starting ElasticSearch client")
    try:
        es = Elasticsearch([conn_string])
    except:
        raise ConnectionError(f"Couldn't connect to Elastic Search instance at: {conn_string} \
                                Check if you've started it or if it listens on the port listed above.")
    logger.debug("Elasticsearch connected")
    return es

def clear_index(es):
    cleared = True
    try:
        es.indices.delete(index=INDEX_NAME)
    except:
        cleared = False
    return cleared

def build_index(es, vclaims, fieldnames):
    vclaims_count = vclaims.shape[0]
    clear_index(es)
    logger.info(f"Builing index of {vclaims_count} vclaims with fieldnames: {fieldnames}")
    for i, vclaim in tqdm(vclaims.iterrows(), total=vclaims_count):
        if not es.exists(index=INDEX_NAME, id=i):
            body = vclaim.loc[fieldnames].to_dict()
            es.create(index=INDEX_NAME, id=i, body=body)


def get_similarity(tweet,result):
    #print(tweet)
    #print(result)
    # Corpus with example sentences
    corpus = [str(tweet)]

    corpus_embeddings = embedder.encode(corpus)

    queries = [str(result)]
    query_embeddings = embedder.encode(queries)


    for query, query_embedding in zip(queries, query_embeddings):
        #print (len(query_embedding))
        distances = 1-scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        #print(distances[0])
        if (distances[0]>0.55):
            return 25
        
        else:
            return 0
            


def get_score(es, tweet, search_keys, size=10000):
    query = {"query": {"multi_match": {"query": tweet, "fields": search_keys}}}
    try:
        response = es.search(index=INDEX_NAME, body=query, size=size)
    except:
        logger.error(f"No elasticsearch results for {tweet}")
        raise

    results = response['hits']['hits']
    upp_limit=(len(results))
    
    if upp_limit>1000:
        upp_limit=1000
    #print (upp_limit)
    
    #print (results[0])
    #print("########################################")
    #print(tweet)
    for i in range(0,upp_limit):
        try:
            #print (tweet)
            #print (result['_source']['vclaim'])
            #print(results[i]['_score'])
            results[i]['_score']=results[i]['_score'] + get_similarity(str(tweet),str(results[i]['_source']['vclaim'])) + get_overlap(str(tweet), str(results[i]['_source']['vclaim'] + ' ' + results[i]['_source']['title'] ))
            #print(results[i]['_score'])
            info = results[i].pop('_source')
            results[i].update(info)
        except:
            pass
            
    df = pd.DataFrame(results[0:upp_limit])
    df['id'] = df._id.astype('string').values
    df = df.set_index('id')
    df=df.sort_values(by=['_score'], ascending=False)
    #print(df._score)
    return df._score

def get_scores(es, tweets, vclaims, search_keys, size):
    tweets_count, vclaims_count = len(tweets), len(vclaims)
    scores = {}

    logger.info(f"Geting RM5 scores for {tweets_count} tweets and {vclaims_count} vclaims")
    for i, tweet in tqdm(tweets.iterrows(), total=tweets_count):
        print("1")
        score = get_score(es, tweet.tweet_content, search_keys=search_keys, size=size)
        scores[i] = score
        #print (tweet)
        #print (score)
    return scores

def format_scores(scores):
    formatted_scores = []
    for tweet_id, s in scores.items():
        for vclaim_id, score in s.items():
            row = (str(tweet_id), 'Q0', str(vclaim_id), '1', str(score), 'elasic')
            formatted_scores.append(row)
    formatted_scores_df = pd.DataFrame(formatted_scores, columns=PREDICT_FILE_COLUMNS)
    return formatted_scores_df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vclaims", "-v", required=True,
                        help="TSV file with vclaims. Format: vclaim_id vclaim title")
    parser.add_argument("--tweets", "-t", required=True,
                        help="TSV file with tweets. Format: tweet_id tweet_content")
    parser.add_argument("--predict-file", "-p", required=True,
                        help="File in TREC Run format containing the model predictions")
    parser.add_argument("--keys", "-k", default=['vclaim', 'title'],
                        help="Keys to search in the document")
    parser.add_argument("--size", "-s", default=10000,
                        help="Maximum results extracted for a query")
    parser.add_argument("--conn", "-c", default="127.0.0.1:9200",
                        help="HTTP/S URI to a instance of ElasticSearch")
    return parser.parse_args()

def main(args):
    vclaims = pd.read_csv(args.vclaims, sep='\t', index_col=0)
    tweets = pd.read_csv(args.tweets, sep='\t', index_col=0)

    es = create_connection(args.conn)
    #es = Elasticsearch(

    #cloud_id="checkthat:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDk1NjE3MTA1MDYzODRmNzViMDFlNzVlZTQyODhlZTBlJDI5MDRmYmYzOWVhMDRhY2NiYTRlODE4MGI3NjBjZTJj",
    #http_auth=("elastic", "bXy788iqxs0prhGDHhToucCn"),)

    build_index(es, vclaims, fieldnames=args.keys)
    scores = get_scores(es, tweets, vclaims, search_keys=args.keys, size=args.size)
    #clear_index(es)

    formatted_scores = format_scores(scores)
    formatted_scores.to_csv(args.predict_file, sep='\t', index=False, header=False)
    logger.info(f"Saved scores from the model in file: {args.predict_file}")

if __name__=='__main__':
    args = parse_args()
    main(args)
