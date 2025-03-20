import re
import pandas as pd
import numpy as np
import nltk
import nltk
nltk.data.path.append('/Users/ilknurakcay/Desktop/rec_sys')
# Şimdi 'punkt' veri setini kullanabilirsiniz
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are downloaded
def download_nltk_resources():
    try:
        nltk.download('stopwords')
    except LookupError:
        print("Downloading additional resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        

def preprocess_text(text):
    """
    Preprocesses a given text by converting it to lowercase, removing special characters and numbers, 
    tokenizing it, and filtering out Turkish stop words. Finally, it returns the cleaned text as a string.
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase and Remove special characters and numbers
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('turkish'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a string
    return ' '.join(tokens)


#If you want to train model, you can use folowing 2  lines and and run create_embeddings_subgraph.
#model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Türkçe desteği olan çok dilli model
#sentence_transformer = SentenceTransformer(model_name)
"""
def create_embeddings_subgraph(item_info, sentence_transformer, G_subgraph):
   
   # Generates text embeddings for job nodes that exist in the given subgraph.

    #Parameters:
    #- item_info (pd.DataFrame): A dataframe containing job-related information.
    #- sentence_transformer (SentenceTransformer): A pre-trained sentence embedding model used to generate embeddings.
    #- G_subgraph: A subgraph containing job and user nodes.

    
    item_embeddings = {}
    for idx, row in item_info.iterrows():
      item_id = row['item_id']
      job_node = f"J_{item_id}"
      if job_node in G_subgraph.nodes():
        text = row['combined_text']
        embedding = sentence_transformer.encode(text)
        item_embeddings[job_node] = embedding
        #print("item_embeddings[job_node]",item_embeddings[job_node])
    return item_embeddings
"""



if __name__ == "__main__":
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    item_info_path = "/Users/ilknurakcay/Desktop/rec_sys/item_information.csv"
    item_info = pd.read_csv(item_info_path)
    
    # Load subgraph
    with open("job_subgraph.pkl", "rb") as f:
        G_subgraph = pickle.load(f)
    
    # Prepare text data
    print("Preprocessing text data...")
    item_info['clean_title'] = item_info['pozisyon_adi'].apply(preprocess_text)
    item_info['clean_description'] = item_info['item_id_aciklama'].apply(preprocess_text)
    # Combine title and description for each item
    item_info['combined_text'] = item_info['clean_title'] + ' ' + item_info['clean_description']

    print(item_info[['item_id_aciklama', 'clean_description']].head(2))


    # create_embeddings_subgraph(item_info, sentence_transformer, G_subgraph):


    #I trained this model before so I just upload.
    with open("/Users/ilknurakcay/Desktop/rec_sys/item_embeddings.pkl", "rb") as f:
      loaded_item_embeddings = pickle.load(f)
    print("SentenceTransformer Model successfuly loaded!")

