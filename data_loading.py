import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(item_info_path, user_data_path):
    """
    Load the dataset files.
    
    Parameters:
    - item_info_path: item information CSV file
    - user_data_path: user event data CSV file
    
    """
    item_info = pd.read_csv(item_info_path)  
    user_data = pd.read_csv(user_data_path) 
    
    print(f"Loaded {len(item_info)} job listings and {len(user_data)} user interactions")
    return item_info, user_data


def create_graph(item_info, user_data):
    """
    Creates graph  between users and job postings.

    Parameters:
    - item_info (pandas.DataFrame): A dataframe containing job details.
      Required columns: 
      - 'item_id' (job ID)
      - 'pozisyon_adi' (job title/position)
      - 'item_id_aciklama' (job description)
    
    - user_data (pandas.DataFrame): A dataframe containing user interactions with jobs.
      Required columns:
      - 'client_id' (user ID)
      - 'item_id' (job ID the user interacted with)
      - 'event_type' (interaction type: 'click' or 'purchase')
      - 'ds_search_id' (search session ID)

    """
    G = nx.Graph()

    #client ids/user and item ids/job
    for _, row in user_data.iterrows():
        user = f"U_{row['client_id']}"  
        job = f"J_{row['item_id']}"  

        # User and job nodes added graph
        G.add_node(user, type="user")
        G.add_node(job, type="job")

        #Edge Type: click → 1, purchase → 3
        weight = 1 if row['event_type'] == 'click' else 3

        #Edges addes with User-job 
        G.add_edge(user, job, weight=weight)

        # Labels and descriptions added to jobs
        job_info = item_info[item_info['item_id'] == row['item_id']].iloc[0]  
        G.nodes[job]['label'] = job_info['pozisyon_adi']  # Label
        G.nodes[job]['description'] = job_info['item_id_aciklama']  #Descriptions

    # Link job appearing in the same search (by ds_search_id)
    search_groups = user_data.groupby("ds_search_id")["item_id"].apply(list)
    for items in search_groups:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                job1 = f"J_{items[i]}"
                job2 = f"J_{items[j]}"
                if not G.has_edge(job1, job2):
                    G.add_edge(job1, job2, weight=0.5)  # If 2 jobs appears in same search, weight is 0.5

    # Check the grap content (Unique Node Type)
    unique_types = set()  

    print("Graph Nodes with Unique Types:")
    for node, data in G.nodes(data=True):
        node_type = data.get("type")
        if node_type and node_type not in unique_types:
            print(f"Type: {node_type}")
            unique_types.add(node_type) 


    return G


def get_subgraph(G, num_nodes=10000):
    """
    Extracts a subgraph from the given graph by selecting a limited number of nodes.

    Parameters:
    - G : The original graph.
    - num_nodes : The number of nodes to include in the subgraph. It was determined as 10000 due to lack of resources

    """
    nodes_sample = list(G.nodes())[:num_nodes]  
    subgraph = G.subgraph(nodes_sample)
    print(f"Subgraph created.Nodes numbers: {len(subgraph.nodes())}")
    return subgraph


if __name__ == "__main__":
    item_info_path = "/Users/ilknurakcay/Desktop/rec_sys/item_information.csv"
    user_data_path = "/Users/ilknurakcay/Desktop/rec_sys/user_event_data.csv"


    # Load data
    print("Loading data...")
    item_info, user_data = load_data(item_info_path, user_data_path)
    
    # Create graph
    print("Creating graph...")
    G = create_graph(item_info, user_data)
    
    # Create subgraph
    print("Creating subgraph...")
    G_subgraph = get_subgraph(G)
    
    # Save graph for later use
    import pickle

    with open("job_subgraph.pkl", "wb") as f:
        pickle.dump(G_subgraph, f)
    
    print("Subgraph saved to disk.")