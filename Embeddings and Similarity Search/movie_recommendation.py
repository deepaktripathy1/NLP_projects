import pandas as pd
import json5
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def data_prep():
    movie = pd.read_csv(r"./movie_recommendation/movie_database/movies_metadata.csv", dtype="object")
    # select relevant columns
    movie = movie[["id", "original_title", "title", "genres"]]

    # using regex to remove values with date format from id column
    def remove_date_format(text):
        text = re.sub(r'(19|20)\d\d[-](0[1-9]|1[012])[-](0[1-9]|[12][0-9]|[3][01])', 'NA', text)
        return text

    # apply function
    movie["id"] = movie["id"].apply(lambda x: remove_date_format(x))
    # filter out "NA" entries from id column
    movie = movie[movie["id"] != "NA"]
    # convert id column to numeric
    movie["id"] = pd.to_numeric(movie["id"])
    # load keyword dataset and join with movie dataframe
    keyword = pd.read_csv(r"./movie_recommendation/movie_database/keywords.csv")
    df = movie.join(keyword, how="inner", lsuffix="1", rsuffix="2")

    # create function to convert json strings to text
    def convert_genres_keyword(row):
        genre = json5.loads(row["genres"])
        genre = " ".join(["".join(i["name"].split()) for i in genre])
        key_word = json5.loads(row["keywords"])
        key_word = " ".join(["".join(i["name"].split()) for i in key_word])
        return "%s %s" % (genre, key_word)

    # create a new column with genre and keyword combination
    df["string"] = df.apply(convert_genres_keyword, axis=1)
    return df


def main():
    df = data_prep()
    # create tfidf object
    tfidf = TfidfVectorizer()
    model = tfidf.fit_transform(df["string"])
    # mapping movie title to index
    movie2index = pd.Series(df.index, index=df["title"])
    title = input("Enter the movie title:   ")
    idx = movie2index[str(title)]
    if type(idx) == pd.Series:
        idx = idx.iloc[0]
    # get pairwise similarity
    query = model[idx]
    query.toarray()
    scores = cosine_similarity(query, model)
    scores = scores.flatten()
    # get indices of top 5 matches
    recommended = (-scores).argsort()[1:6]
    titles = df['title'].iloc[recommended]
    return print(list(titles.values))


if __name__ == "__main__":
    main()
