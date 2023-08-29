from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string
from tqdm import tqdm
import time


def get_urls_from_sitemap(resource_url: str) -> list:
    """
    Get a list of urls from sitemap
    """
    urls = sitemap_search(resource_url)[:50]
    return urls


def extract_article(url: str) -> dict:
    """
    Extract article from a url
    """
    downloaded = fetch_url(url)
    article = extract(downloaded, favor_precision=True)
    return article


def create_dataset(website):
    """
    Create a dataframe from a list of sitemaps
    that is passed to get_urls_from_sitemap

    """
    data = []
    urls = get_urls_from_sitemap(website)
    # create a dictionary
    for url in tqdm(urls, desc="URLs"):
        d = {
            "url": url,
            "article": extract_article(url)
        }
        data.append(d)
        time.sleep(0.5)  # this ensures not over pinging the website

    df = pd.DataFrame(data)
    # remove duplicates
    df = df.drop_duplicates()
    # remove empty rows
    df = df.dropna()

    blogs = df[df["url"].str.contains("blog")]  # retain urls containing blog
    blogs.reset_index(drop=True, inplace=True)

    return blogs


def preprocess_data(text):
    nltk.download('stopwords')
    stop_words = list(stopwords.words('english'))  # create stopwords list
    # lowercase
    text = text.lower()
    # remove stopwords
    text = ' '.join([t for t in text.split() if t not in stop_words])
    # remove non-ascii
    text = ''.join(word for word in text if ord(word) < 128)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text


def main():
    website = input("Enter website link:  ")
    blogs = create_dataset(str(website))
    blogs["article_final"] = blogs["article"].apply(lambda x: preprocess_data(x))
    # extract title from url
    blogs["title"] = blogs["url"].str.split('/').str[6:].str[1]
    # create tfidf object and transform text
    tfidf = TfidfVectorizer()
    model = tfidf.fit_transform(blogs["article_final"])
    # compute cosine similarity
    M = np.zeros((blogs.shape[0], blogs.shape[0]))  # create empty 43 x 43 matrix to store similarity scores
    for i in range(len(blogs)):
        for j in range(len(blogs)):
            query_i = model[i].toarray()
            query_j = model[j].toarray()
            M[i, j] = cosine_similarity(query_i, query_j)
    # Visualize similarity matrix using a heatmap
    similarity_df = pd.DataFrame(M, columns=blogs.title.values, index=blogs.title.values)
    mask = np.triu(np.ones_like(similarity_df))  # create a mask to remove top of the dataframe
    # plot
    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes()
    ax.set_facecolor("black")
    sns.heatmap(similarity_df,
                square=True,
                annot=True,
                fmt=".2f",
                xticklabels=similarity_df.columns,
                yticklabels=similarity_df.columns,
                cmap="YlGnBu",
                mask=mask)

    plt.title("Similarity Heatmap", fontdict={"fontsize": 22})
    plt.show()


if __name__ == "__main__":
    main()
