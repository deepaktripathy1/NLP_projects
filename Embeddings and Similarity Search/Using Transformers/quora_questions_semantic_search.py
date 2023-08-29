# -*- coding: utf-8 -*-
"""quora_questions_semantic_search.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-Yju21A6zUmGcChB1sQHO7PD32UES5Nv
"""

!pip install sentence-transformers

from torch.cuda import is_available
from sentence_transformers import SentenceTransformer, util
import os
import time
import csv
import torch

if torch.cuda.is_available():
  print("GPU available and ready to go")

model_name = "quora-distilbert-multilingual"
model = SentenceTransformer(model_name)

url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 100000

if not os.path.exists(dataset_path):
    print("Download dataset")
    util.http_get(url, dataset_path)

import pandas as pd

df = pd.read_csv(r'./drive/MyDrive/quora_duplicate_questions.tsv',sep='\t')
df.head()

len(df)

df['question1'].nunique()

df['question2'].nunique()

# Alternate wat to extract data

#corpus_sentences = set()
#with open(dataset_path, encoding='utf8') as fIn:
    #reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    #for row in reader:
        #corpus_sentences.add(row['question1'])
        #if len(corpus_sentences) >= max_corpus_size:
            #break

        #corpus_sentences.add(row['question2'])
        #if len(corpus_sentences) >= max_corpus_size:
            #break

# Get all unique statements from the file
corpus_sentences = set()
for sentence1,sentence2 in zip(df['question1'].values,df['question2'].values):
  corpus_sentences.add(sentence1)
  corpus_sentences.add(sentence2)
  if len(corpus_sentences) >= max_corpus_size:
    break

corpus_sentences = list(corpus_sentences)
len(corpus_sentences)

# Encode the corpus statements

embeddings = model.encode(corpus_sentences,
                          show_progress_bar=True,
                          convert_to_tensor=True)

print("Shape of embeddings:",embeddings.shape)

# Function that searches the corpus and prints results

def search(input_ques):
  start_time = time.time()
  ques_embedding = model.encode(input_ques,convert_to_tensor=True)
  hits = util.semantic_search(ques_embedding,embeddings)
  end_time = time.time()
  hits = hits[0] # get hits for the first query

  print("Input question:",input_ques)
  print("Results after {:.3f} seconds:".format(end_time-start_time))
  for hit in hits[:5]:
    print("\t{:.3f}\t{}".format(hit['score'],
                                corpus_sentences[hit['corpus_id']]))

# test the function

search("How can I learn python online")

search("What are the best tourist spots in Australia")

#German: How can I learn Python online?
search("Wie kann ich Python online lernen?")

