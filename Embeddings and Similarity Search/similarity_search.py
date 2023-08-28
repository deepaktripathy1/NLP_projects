import spacy
from scipy import spatial
nlp = spacy.load("en_core_web_lg")

def similarity_search(word):
    cosine_similarity = lambda x,y: 1- spatial.distance.cosine(x,y)
    vector = nlp.vocab[str(word)].vector
    computed_similarities = {}
    for word in nlp.vocab:
        if word.has_vector:
            if word.is_alpha:
                if word.is_lower:
                    if not word.is_stop:
                        similarity = cosine_similarity(vector,word.vector)
                        computed_similarities[word.text] = similarity
    return computed_similarities
    

def main():
    word = input("Enter the word :")
    computed_similarities = similarity_search(word)
    print(sorted(computed_similarities.items(),key=lambda x:x[1],reverse=True)[:5])
    
if __name__ == "__main__":
    main()