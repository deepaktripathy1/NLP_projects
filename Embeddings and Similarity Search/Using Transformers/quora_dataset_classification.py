# -*- coding: utf-8 -*-
"""quora_dataset_classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19QSbpNdLjBdazVTq64B016bRfJnTUXu9
"""

!pip install sentence-transformers

from torch.utils.data import DataLoader
import math
from sentence_transformers import util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
from datetime import datetime
import sys
import os
import gzip
import csv
import pandas as pd
from zipfile import ZipFile

#Check if dataset exsist. If not, download and extract  it
dataset_path = 'quora-dataset/'

if not os.path.exists(dataset_path):
  zip_save_path = 'quora-IR-dataset.zip'
  util.http_get(url='https://sbert.net/datasets/quora-IR-dataset.zip',
                path = zip_save_path)
  with ZipFile(zip_save_path,'r') as zip:
    zip.extractall(dataset_path)

col_names = ['qid1','qid2','question1','question2','is_duplicate']
df_train = pd.read_csv(r'./quora-dataset/classification/train_pairs.tsv',
                       sep = '\t',
                       encoding='utf8',
                       on_bad_lines='skip')
df_train.head()

len(df_train)

df_train.isnull().sum()

# drop null values
df_train.dropna(inplace=True)
df_train.reset_index(drop=True,inplace=True)
len(df_train)

train_samples = []
for i in range(len(df_train[:50000])):
  train_samples.append(InputExample(texts=[df_train.loc[i,'question1'],df_train.loc[i,'question2']],
                                    label = int(df_train.loc[i,'is_duplicate'])))

df_dev = pd.read_csv(r'./quora-dataset/classification/dev_pairs.tsv',
                       sep = '\t',
                       encoding='utf8',
                       on_bad_lines='skip')
df_dev.head()

len(df_dev)

dev_samples = []
for i in range(len(df_dev)):
  dev_samples.append(InputExample(texts=[df_dev.loc[i,'question1'],df_dev.loc[i,'question2']],
                                    label = int(df_dev.loc[i,'is_duplicate'])))

#Configuration
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_quora-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#We use distilroberta-base with a single label, i.e., it will output a value between 0 and 1 indicating the similarity of the two questions
model = CrossEncoder('distilroberta-base', num_labels=1)

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, 
                                                                name='Quora-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

df_test = pd.read_csv(r'./quora-dataset/classification/test_pairs.tsv',
                       sep = '\t',
                       encoding='utf8',
                       on_bad_lines='skip')
df_test.head()

test_samples = []
for i in range(len(df_test)):
  test_samples.append(InputExample(texts=[df_test.loc[i,'question1'],df_test.loc[i,'question2']],
                                    label = int(df_test.loc[i,'is_duplicate'])))

len(test_samples)

##### Load model and eval on test set
model = CrossEncoder(model_save_path)

evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples,name='Quora-test')
evaluator(model)

scores_epochs = pd.read_csv(r'/content/output/training_quora-2023-06-02_17-05-16/CEBinaryClassificationEvaluator_Quora-dev_results.csv')
scores_epochs

test = []
for i in range(len(df_test)):
  test.append([df_test.loc[i,'question1'],df_test.loc[i,'question2']])

predictions = model.predict(test,show_progress_bar=True)
predictions

cos_sim = list(predictions)
df_test['cosine_sim'] = cos_sim
df_test.head()

# filter dataframes with duplicate == Yes
df_dup_1 = df_test[df_test['is_duplicate']==1]
df_dup_1.head()

# filter dataframe with duplicate == No
df_dup_0 = df_test[df_test['is_duplicate']==0]
df_dup_0.head()

# Histogram
import matplotlib.pyplot as plt
import seaborn as sns

fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))
fig.tight_layout(pad=3)
sns.histplot(df_dup_0['cosine_sim'],kde=False,ax=ax1)
sns.histplot(df_dup_1['cosine_sim'],kde=False,ax=ax2)
ax1.set_title('Duplicates = No')
ax2.set_title('Duplicates = Yes')
plt.show()

df_pred = df_test[['cosine_sim','is_duplicate']]
df_pred.head()

X = df_pred[['cosine_sim']]
y = df_pred['is_duplicate']

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

rfc = RandomForestClassifier(n_estimators=300)
model_rfc = rfc.fit(X,y)
y_pred = model_rfc.predict(X)
y_pred

print('Accuracy Score:',accuracy_score(y,y_pred))

print('Classification Score:',classification_report(y,y_pred))

# Read the quora dataset split for classification

train_samples = []
with open(os.path.join(dataset_path,'classification','train_pairs.tsv'),'r',encoding='utf8') as f:
  reader = csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
  for row in reader:
    train_samples.append(InputExample(texts=[row['question1'],row['question2']],
                                      label = int(row['is_duplicate'])))
    train_samples.append(InputExample(texts=[row['question2'],row['question1']],
                                      label = int(row['is_duplicate'])))
    
dev_samples = []
with open(os.path.join(dataset_path,'classification','dev_pairs.tsv'),'r',encoding='utf8') as f:
  reader = csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
  for row in reader:
    dev_samples.append(InputExample(texts=[row['question1'],row['question2']],
                                      label = int(row['is_duplicate'])))
    
#Configuration
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_quora-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#We use distilroberta-base with a single label, i.e., it will output a value between 0 and 1 indicating the similarity of the two questions
model = CrossEncoder('distilroberta-base', num_labels=1)

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, 
                                                                name='Quora-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

test_samples = []
with open(os.path.join(dataset_path,'classification','test_pairs.tsv'),'r',encoding='utf8') as f:
  reader = csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
  for row in reader:
    test_samples.append(InputExample(texts=[row['question1'],row['question2']],
                                      label = int(row['is_duplicate'])))
    
##### Load model and eval on test set
model = CrossEncoder(model_save_path)

evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples,name='Quora-test')
evaluator(model)