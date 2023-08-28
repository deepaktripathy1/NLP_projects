# -*- coding: utf-8 -*-
"""retrieve_rerank_simple_wikipedia.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1344plX8F5A9wYgWuujvkRew0H3BP0HVz

Retrieve & Re-Rank Demo over Simple Wikipedia
This examples demonstrates the Retrieve & Re-Rank Setup and allows to search over Simple Wikipedia.

You can input a query or a question. The script then uses semantic search to find relevant passages in Simple English Wikipedia (as it is smaller and fits better in RAM).

For semantic search, we use SentenceTransformer('multi-qa-MiniLM-L6-cos-v1') and retrieve 32 potentially passages that answer the input query.

Next, we use a more powerful CrossEncoder (cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')) that scores the query and all retrieved passages for their relevancy. The cross-encoder further boost the performance, especially when you search over a corpus for which the bi-encoder was not trained for.
"""

!pip install -U sentence-transformers rank_bm25

import json
import gzip
import os
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder

if torch.cuda.is_available():
  print("GPU available and ready to go")

# Bi-encoder
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256 # truncate long passages to 256 tokens
top_k = 32 # Number of passages we want to retrieve

# the bi-encoder will  retrieve 100 docs. We use a crossencoder to
# re-rank the results list
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'

if not os.path.exists(wikipedia_filepath):
  util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz',
                wikipedia_filepath)

passages = []
with gzip.open(wikipedia_filepath,'rt',encoding='utf8') as f:
  for line in f:
    data = json.loads(line.strip())
    # Add the first paragraph
    passages.append(data['paragraphs'][0])

print('Passages:',len(passages))

# Encode the passages

corpus_embeddings = bi_encoder.encode(passages,
                                      convert_to_tensor=True,
                                      show_progress_bar=True)

print("Shape of embeddings:",corpus_embeddings.shape)

# We also compare the results to lexical search (keyword search). Here, we use 
# the BM25 algorithm which is implemented in the rank_bm25 package.

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np

# lower case text and remove stopwords

def bm25_tokenizer(text):
  tokenized_doc = []
  for token in text.lower().split():
    token = token.strip(string.punctuation)

    if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
      tokenized_doc.append(token)
  return tokenized_doc

tokenized_corpus = []
for passage in tqdm(passages):
  tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

# This function will search all wikipedia articles for passages that
# answer the query  \ h 
def search(query):
    print("Input question:", query)

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -5)[-5:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    print("Top-3 lexical search (BM25) hits")
    for hit in bm25_hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
      hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))

search(query = "What is the capital of the United States?")

search(query = "When did the cold war end?")

search(query = "Indira Gandhi")

search(query="Dalai Lama")
