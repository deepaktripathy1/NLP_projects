# -*- coding: utf-8 -*-
"""semantic_asymmetric_search_wikipedia_qa.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QbZpBJPsApFF69mIJkWA430YeEVsliVm
"""

!pip install -U sentence-transformers

import json
import os
import time
import gzip
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder

if torch.cuda.is_available():
  print("GPU available and ready to go")

# Create bi-encoder model to encode all the passages
model_name = 'nq-distilbert-base-v1'
bi_encoder = SentenceTransformer(model_name)
top_k = 5 # No.of passages we want to retrive with the bi-encoder

wikipedia_filepath = 'data/simplewiki-2020-11-01.jsonl.gz'

if not os.path.exists(wikipedia_filepath):
  util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz',
                wikipedia_filepath)

passages= []
with gzip.open(wikipedia_filepath,"rt",encoding="utf-8") as f:
  for line in f:
    data = json.loads(line.strip())
    for paragraph in data['paragraphs']:
      # We encode the passages as [title,text]
      passages.append([data['title'],paragraph])

print("Passages:",len(passages))

passages[:2]

# To speed things up, pre-computed embeddings are downloaded.
# The provided file encoded the passages with the model 'nq-distilbert-base-v1'

if model_name == 'nq-distilbert-base-v1':
  embeddings_filepath = 'simplewiki-2020-11-01-nq-distilbert-base-v1.pt'
  if not os.path.exists(embeddings_filepath):
    util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01-nq-distilbert-base-v1.pt',
                  embeddings_filepath)
    
  corpus_embeddings = torch.load(embeddings_filepath)
  corpus_embeddings = corpus_embeddings.float() # convert to float
  if torch.cuda.is_available():
    corpus_embeddings = corpus_embeddings.to('cuda')
else:
  corpus_embeddings = bi_encoder.encode(passages,
                                        convert_to_tensor=True,
                                        show_progress_bar=True)

def search(query):
  start_time = time.time()
  ques_embedding = bi_encoder.encode(query,convert_to_tensor=True)
  hits = util.semantic_search(ques_embedding,corpus_embeddings,top_k=top_k)
  hits = hits[0] # get hits for the first query
  end_time = time.time()

  # Output of top-k hits
  print("Input Question:",query)
  print("Results after {:.3f} seconds:".format(end_time - start_time))
  for hit in hits:
    print("\t{:.3f}\t{}".format(hit['score'],passages[hit['corpus_id']]))

search(query = "What is the capital of the France?")

search("When was USA founded?")

search("Who was Ashoka the Great?")

search("What is the history of Ukraine?")

search("Grey's Anatomy")

search("Lost TV Series")

from sentence_transformers import SentenceTransformer,util
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

model = SentenceTransformer('nq-distilbert-base-v1')

query_embedding = model.encode('How many people live in London?',
                               convert_to_numpy=True)
# OR
query_embed = model.encode('How many people live in London?')

# Encode passages as [title,text]
passage_embedding = model.encode([['London','London has 9,787,426 inhabitants at the 2011 census.']],
                                 convert_to_numpy=True)

# OR

passage_embed = model.encode([['London','London has 9,787,426 inhabitants at the 2011 census.']])

print('Similarity:',1 - cosine(query_embedding.reshape(-1),
                               passage_embedding.reshape(-1)))

print('Cosine Similarity:',cosine_similarity(query_embedding.reshape(1,-1),
                                             passage_embedding.reshape(1,-1))[0])

print('Pytorch similarity:',util.pytorch_cos_sim(query_embed,
                                                 passage_embed))

query_embedding = model.encode('who turned out to be the mother on how i met your mother')

#The passages are encoded as [title, text]
passage_embedding = model.encode([['The Mother (How I Met Your Mother)', 'The Mother (How I Met Your Mother) Tracy McConnell (colloquial: "The Mother") is the title character from the CBS television sitcom "How I Met Your Mother". The show, narrated by Future Ted (Bob Saget), tells the story of how Ted Mosby (Josh Radnor) met The Mother. Tracy McConnell appears in eight episodes, from "Lucky Penny" to "The Time Travelers", as an unseen character; she was first seen fully in "Something New" and was promoted to a main character in season 9. The Mother is played by Cristin Milioti. The story of how Ted met The Mother is the framing device'],
                                  ['Make It Easy on Me', 'and Pete Waterman on her 1993 album "Good \'N\' Ready", on which a remixed version of the song is included. "Make It Easy On Me", a mid-tempo R&B jam, received good reviews (especially for signalling a different, more soulful and mature sound atypical of the producers\' Europop fare), but failed to make an impact on the charts, barely making the UK top 100 peaking at #99, and peaking at #52 on the "Billboard" R&B charts. The pop group Steps covered the song on their 1999 album "Steptacular". It was sung as a solo by Lisa Scott-Lee. Make It Easy on']])

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))

query_embedding = model.encode('where does the story the great gatsby take place')
passage_embedding = model.encode([['The Great Gatsby', 
 'The Great Gatsby The Great Gatsby is a 1925 novel written by American author F. Scott Fitzgerald that follows a cast of characters living in the fictional towns of West Egg and East Egg on prosperous Long Island in the summer of 1922. The story primarily concerns the young and mysterious millionaire Jay Gatsby and his quixotic passion and obsession with the beautiful former debutante Daisy Buchanan. Considered to be Fitzgerald\'s magnum opus, "The Great Gatsby" explores themes of decadence, idealism, resistance to change, social upheaval, and excess, creating a portrait of the Roaring Twenties that has been described as'],
 ['The Producers (1967 film)', '2005 (to coincide with the remake released that year). In 2011, MGM licensed the title to Shout! Factory to release a DVD and Blu-ray combo pack with new HD transfers and bonus materials. StudioCanal (worldwide rights holder to all of the Embassy Pictures library) released several R2 DVD editions and Blu-ray B releases using a transfer slightly different from the North Ameri can DVD and BDs. The Producers (1967 film) The Producers is a 1967 American satirical comedy film written and directed by Mel Brooks and starring Zero Mostel, Gene Wilder, Dick Shawn, and Kenneth Mars. The film was Brooks\'s directorial']
])

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
