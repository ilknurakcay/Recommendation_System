import faiss
import numpy as np
from gensim.models import Word2Vec
import pickle
import pandas as pd

def combine_embeddings(loaded_item_embeddings, model, vector_size=64):
    """
    It creates a final_embeddings list by combining the Node2Vec and Job Embeddings vectors of all job IDs.
    Parameters:
    - loaded_item_embeddings:  containing job embeddings (ID -> 384 dimensional vector).
    - model: Model containing Node2Vec vectors.
    - vector_size: Size of Node2Vec vectors (default: 64).

    Return:
    - final_embeddings: List of 448-dimensional embedding vectors.  
    """
    final_embeddings = []
    item_ids = loaded_item_embeddings.keys() 

    for item_id in item_ids:
        formatted_item_id = str(item_id) 
        if not formatted_item_id.startswith("J_"):
            formatted_item_id = f"J_{formatted_item_id}"  

        if formatted_item_id in model.wv:
            node_vec = model.wv[formatted_item_id]  # Node2Vec vector for jobs (64 dim)
            #print(f"Vector found for {formatted_item_id}!")  
        else:
            node_vec = np.zeros(vector_size) 
            #print(f"WARNING: Node2Vec vector not found for {formatted_item_id}!") 

        job_vec = loaded_item_embeddings[item_id]  # Job Embedding (384 dim) with description and label
        combined_vec = np.concatenate([node_vec, job_vec])  # 448 dim
        final_embeddings.append(combined_vec)

    return final_embeddings, item_ids



def create_faiss_index(final_embeddings, item_ids):
    """
    Creates a FAISS index using the given embeddings and maps item IDs to index positions.

    Parameters:
    - final_embeddings: List or array of job embedding vectors (shape: [N, 448]).
    - item_ids: List of item IDs corresponding to the embeddings.

    """
    final_embeddings = np.array(final_embeddings).astype(np.float32)  # Ensure correct dtype

    dim = final_embeddings.shape[1]  # Get embedding dimension (e.g., 448)
    index = faiss.IndexFlatL2(dim)  # Create L2 FAISS index

    index.add(final_embeddings)  # Add embeddings to FAISS index

    # Create a mapping from item_id to FAISS index position
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    return index, item_id_to_index


def get_top_n_similar_jobs(job_id, N=5):
    """
    Returns the top N most similar job listings based on the specified job ID using FAISS.
    
    Parameters:
        job_id : The job ID for which recommendations are requested.
        N (int): The number of similar job listings to return.
    """
    #Check the format
    formatted_job_id = str(job_id)
    if not formatted_job_id.startswith("J_"):
        formatted_job_id = f"J_{formatted_job_id}"

    #If the job is not in the database, give an error
    if formatted_job_id not in item_id_to_index:
        print(f"WARNING: {formatted_job_id} vector is not find database!")
        return []

    # Get the index of the corresponding vector at the FAISS index
    job_index = item_id_to_index[formatted_job_id]
    job_vector = final_embeddings[job_index].reshape(1, -1).astype(np.float32)

    # Find N nearest neighbors with FAISS
    distances, indices = index.search(job_vector, N+1)  

    # Process the result (ID, position, similarity)
    similar_jobs = []
    for i in range(1, N+1):
        idx = indices[0][i]
        item_id = list(item_id_to_index.keys())[idx]
        distance = distances[0][i]

        #map item_id to position name from item_info
        numeric_item_id = item_id.lstrip("J_")  # 'J_' önekini kaldırıyoruz
        position = item_info[item_info["item_id"] == int(numeric_item_id)]["pozisyon_adi"].values
        position = position[0] if len(position) > 0 else "Unknown"

        similar_jobs.append((item_id, position, distance))

    return similar_jobs




if __name__ == "__main__":
    item_info_path = "/Users/ilknurakcay/Desktop/rec_sys/item_information.csv"
    item_info = pd.read_csv(item_info_path)

    model = Word2Vec.load("/Users/ilknurakcay/Desktop/rec_sys/node2vec_model.bin", mmap='r')
    with open("/Users/ilknurakcay/Desktop/rec_sys/item_embeddings.pkl", "rb") as f:
      loaded_item_embeddings = pickle.load(f)
    print("Combining embeddings...")
    final_embeddings, item_ids = combine_embeddings(loaded_item_embeddings, model)
    print("Creating FAISS index...")
    index, item_id_to_index = create_faiss_index(final_embeddings, item_ids)
    print(len(final_embeddings))
    print("")


    # Example:
    job_id = "J_4031363"  
    N = 5  # 5 most similar jobs
    similar_jobs = get_top_n_similar_jobs(job_id, N)
    numeric_item_id = job_id.lstrip("J_")
    #Search item_info to get the position
    position = item_info[item_info["item_id"] == int(numeric_item_id)]["pozisyon_adi"].values
    if len(position) > 0:
        print(f"Position of Job ID {job_id}: {position[0]}")
    else:
        print(f"No position information found for {job_id}!")
    print("Most similar jobs:")
    for job, position, score in similar_jobs:
        print(f"{job} - {position} - Similarity Score: {score:.4f}")