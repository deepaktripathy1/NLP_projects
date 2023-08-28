# -*- coding: utf-8 -*-
"""Image_Search-multilingual.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N6woBKL4dzYsHboDNqtv-8gjZglKOZcn

# Multilingual Joint Image & Text Embeddings 

This example shows how [SentenceTransformers](https://www.sbert.net) can be used to map images and texts to the same vector space. 

As model, we use the [OpenAI CLIP Model](https://github.com/openai/CLIP), which was trained on a large set of images and image alt texts.

The original CLIP Model only works for English, hence, we used [Multilingual Knowlegde Distillation](https://arxiv.org/abs/2004.09813) to make this model work with 50+ languages.

As a source for fotos, we use the [Unsplash Dataset Lite](https://unsplash.com/data), which contains about 25k images. See the [License](https://unsplash.com/license) about the Unsplash images. 

Note: 25k images is rather small. If you search for really specific terms, the chance are high that no such photo exist in the collection.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm

# Here we load the multilingual CLIP model. Note, this model can only encode text.
# If you need embeddings for images, you must load the 'clip-ViT-B-32' model
model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

# Next, we get about 25k images from Unsplash 
img_folder = 'photos/'
if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
    os.makedirs(img_folder, exist_ok=True)
    
    photo_filename = 'unsplash-25k-photos.zip'
    if not os.path.exists(photo_filename):   #Download dataset if does not exist
        util.http_get('http://sbert.net/datasets/'+photo_filename, photo_filename)
        
    #Extract all images
    with zipfile.ZipFile(photo_filename, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            zf.extract(member, img_folder)

# Now, we need to compute the embeddings
# To speed things up, we destribute pre-computed embeddings
# Otherwise you can also encode the images yourself.
# To encode an image, you can use the following code:
# from PIL import Image
# img_emb = model.encode(Image.open(filepath))

use_precomputed_embeddings = True

if use_precomputed_embeddings: 
    emb_filename = 'unsplash-25k-photos-embeddings.pkl'
    if not os.path.exists(emb_filename):   #Download dataset if does not exist
        util.http_get('http://sbert.net/datasets/'+emb_filename, emb_filename)
        
    with open(emb_filename, 'rb') as fIn:
        img_names, img_emb = pickle.load(fIn)  
    print("Images:", len(img_names))
else:
    #For embedding images, we need the non-multilingual CLIP model
    img_model = SentenceTransformer('clip-ViT-B-32')

    img_names = list(glob.glob('photos/*.jpg'))
    print("Images:", len(img_names))
    img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

import torch
filepath = 'photos/'+img_names[0]
one_emb = torch.tensor(img_emb[0])
img_model = SentenceTransformer('clip-ViT-B-32')
comb_emb = img_model.encode(Image.open(filepath), convert_to_tensor=True).cpu()
print(util.cos_sim(one_emb, comb_emb))

# Next, we define a search function.
def search(query, k=3):
    # First, we encode the query (which can either be an image or a text string)
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
    
    print("Query:")
    display(query)
    for hit in hits:
        print(img_names[hit['corpus_id']])
        display(IPImage(os.path.join(img_folder, img_names[hit['corpus_id']]), width=200))

search("Two dogs playing in the snow")

#German: A cat on a chair
search("Eine Katze auf einem Stuhl")

#Spanish: Many fish
search("Muchos peces")

#Chinese: A beach with palm trees
search("棕榈树的沙滩")

#Russian: A sunset on the beach
search("Закат на пляже")

#Turkish: A dog in a park
search("Parkta bir köpek")

# Japanese: New York at night
search("夜のニューヨーク")