# Job Recommendation System

A hybrid recommendation system for job listings based on user interaction user and job content. This system is graph-based approaches provide relevant job recommendations to users.

## Project Structure

The project consists of four main Python scripts and a report document:

1. `data_loading.py` - Load datasets and create graph structures
2. `node2vec_embeddings.py` - Generate graph-based embeddings using Node2Vec
3. `text_and_embedding.py` - Process text data and create embeddings from job 
4. `recommendation_system.py` - Combine embeddings and provide job recommendations
5. `İlan Öneri Sistemi Raporu.pdf` - Report

## Prerequisites

### Required Libraries

You can install all the required dependencies using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

The requirements include:
- gensim
- pandas==2.2.0
- pandasai
- torch, torchvision, torchaudio
- networkx
- matplotlib
- scikit-learn
- nltk
- node2vec
- sentence-transformers
- faiss-cpu

### Required Data Files

The system requires two CSV files:
- `item_information.csv` - Contains job listing details
- `user_event_data.csv` - Contains user interaction data

## Workflow

Follow these steps to run the recommendation system:

### 1. Data Loading and Graph Creation

Run the `data_loading.py` script to load the dataset files and create a graph structure:

```bash
python data_loading.py
```

This script:
- Loads job listing information and user event data
- Creates a graph representing user-job interactions
- Creates a subgraph for better memory management
- Saves the subgraph as `job_subgraph.pkl`

### 2. Node2Vec Embeddings

Run the `node2vec_embeddings.py` script to generate graph-based embeddings:

```bash
python node2vec_embeddings.py
```

This script:
- Loads the trained Node2Vec model
- Creates job embeddings from the graph structure
- Calculates similarity between random job pairs to evaluate embeddings

### 3. Text Processing and Embedding

Run the `text_and_embedding.py` script to process job descriptions(text-based) embeddings:

```bash
python text_and_embedding.py
```

This script:
- Downloads necessary NLTK resources for text processing
- Preprocesses job titles and descriptions
- Combines cleaned title and description data
- Loads trained SentenceTransformer embeddings 

### 4. Recommendation System

Finally, run the `recommendation_system.py` script to combine embeddings and generate recommendations:

```bash
python recommendation_system.py
```

This script:
- Combines Node2Vec and SentenceTransformer embeddings to create hybrid representations
- Creates a FAISS index for fast similarity search
- Provides a function to get top N similar jobs for any job ID

## Example Usage

To get job recommendations for a specific job listing:

```python
from recommendation_system import get_top_n_similar_jobs

job_id = "J_4031363"  # Example job ID
top_similar_jobs = get_top_n_similar_jobs(job_id, N=5)

for job, position, score in top_similar_jobs:
    print(f"{job} - {position} - Similarity Score: {score:.4f}")
```

### Example Output

When running the recommendation system with job ID J_4031363, you should see output similar to this:

```
Position of Job ID J_4031363: Muhasebe ve Finans Sorumlusu
Most similar jobs:
J_4029881 - Muhasebe Müdürü - Similarity Score: 4.2147
J_4019804 - Muhasebe Uzmanı - Similarity Score: 4.4862
J_4032234 - Müşteri Hizmetleri Müdürü - Similarity Score: 5.3833
J_4035143 - Dönemsel Muhasebe Elemanı - Similarity Score: 5.5619
J_4029367 - Kobi Bankacılığı Portföy Yöneticisi / Ankara - Similarity Score: 5.8603
```

Note that lower similarity scores indicate more relevant job matches.


For more details on the implementation and methodology, refer to the included report document: `İlan Öneri Sistemi Raporu.pdf`.
