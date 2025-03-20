
from gensim.models import Word2Vec
import numpy as np
import itertools
import random
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
    

#If you want to train model, you can use folowing 4  lines.
#node2vec = Node2Vec(graph=G_subgraph, dimensions=64, walk_length=5, num_walks=10, workers=8)
#model = node2vec.fit(window=10, min_count=1, batch_words=4)
# Save to model
#model.save("/content/drive/MyDrive/rec_sys_drive/node2vec_model.bin")

#I trained model before and I will use this.
def load_node2vec_model(model_path):
    """
    Load a pre-trained Node2Vec model.
    """
    model = Word2Vec.load(model_path, mmap='r')
    print("Node2Vec model successfully loaded!")
    return model


def get_job_embeddings(G_subgraph, model):
  """
  This function is generating job vector with user data


  Parameters:
  -G_subgraph: Subgraph containing job and user nodes.
  -model:A trained embedding model that provides vector 
      representations for job nodes.
  """
  user_count = sum(1 for node, data in G_subgraph.nodes(data=True) if data.get("type") == "user")
  job_count = sum(1 for node, data in G_subgraph.nodes(data=True) if data.get("type") == "job")

  print(f"User node number in subgraph: {user_count}")
  print(f"Job node number in subgraph: {job_count}")

  job_embeddings = {}
  for node in G_subgraph.nodes():
    if node.startswith('J_') and node in model.wv:
        job_embeddings[node] = model.wv[node]

  print(f"A total of {len(job_embeddings)} job vectors were created.")
  return job_embeddings




def get_position_name(job_id, item_info):
    """
    Returns the position name corresponding to the given job ID.

    Parameters:
    - job_id (str): The job's ID, e.g., 'J_4031363'.
    - item_info (DataFrame): A pandas DataFrame containing job information.

    """
    item_id = job_id.replace("J_", "")  
    position_name = item_info.loc[item_info['item_id'] == int(item_id), 'pozisyon_adi']
    return position_name.iloc[0] if not position_name.empty else "unknown"

def calculate_average_similarity(job_embeddings, G_subgraph, item_info, num_pairs=5):
    """
    Calculates the similarity between "job" nodes in the subgraph. 
    Cosine similarity is used for the similarity calculation.

    Parameters:
    - job_embeddings : A dictionary containing the embedding vectors for each job. 
      The keys are job IDs, and the values are the embedding vectors.
    - G_subgraph : A subgraph object containing the job and user nodes.
    - item_info :A pandas DataFrame containing job information, including position names.
    - num_pairs : The number of random job pairs to compute. Default is 5.
    """
    similarities = []
    
    #Get "job" nodes in subgraph
    job_nodes = [node for node, data in G_subgraph.nodes(data=True) if data.get("type") == "job"]
    if len(job_nodes) < 2:
        print("There are insufficient job nodes!")
        return None

    # Generate all job pairs and randomly select 10
    all_pairs = list(itertools.combinations(job_nodes, 2))  
    selected_pairs = random.sample(all_pairs, min(len(all_pairs), num_pairs))

    # Get embedding vectors of selected pairs
    job_vectors = {job: job_embeddings[job] for job in job_nodes if job in job_embeddings}

    # Caclulate cosine similarity 
    for job1, job2 in selected_pairs:
        if job1 not in job_vectors or job2 not in job_vectors:
            print(f"WARNING: {job1} or {job2} is not in job_embeddings!")
            continue
        
        # Fast cosine similarity calculation with NumPy matrix multiplication
        vec1 = job_vectors[job1].reshape(1, -1)
        vec2 = job_vectors[job2].reshape(1, -1)
        sim = np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        #Get position name
        pos1 = get_position_name(job1, item_info)
        pos2 = get_position_name(job2, item_info)

        similarities.append((job1, pos1, job2, pos2, sim[0][0]))

    return similarities if similarities else None



if __name__ == "__main__":

    # Path to the pre-trained model
    model_path = "/Users/ilknurakcay/Desktop/rec_sys/node2vec_model.bin"
    # Load the model
    model = load_node2vec_model(model_path)

    # Load the graph and item info
    with open("job_subgraph.pkl", "rb") as f:
        G_subgraph = pickle.load(f)

    item_info_path = "/Users/ilknurakcay/Desktop/rec_sys/item_information.csv"
    item_info = pd.read_csv(item_info_path)
    
    # Get job embeddings
    job_embeddings = get_job_embeddings(G_subgraph, model)
    
    # Calculate similarities for a few random pairs
    similarity_results = calculate_average_similarity(job_embeddings, G_subgraph, item_info, num_pairs=5)
    
    if similarity_results:
        print("\nSample Job Similarities:")
        for job1, pos1, job2, pos2, sim in similarity_results:
            print(f"{job1} ({pos1}) ↔ {job2} ({pos2}) -> Cosine Similarity: {sim:.4f}")
    else:
        print("Similarity was not computed!")