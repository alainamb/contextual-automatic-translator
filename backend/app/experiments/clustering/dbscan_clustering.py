"""
DBSCAN Clustering Test for Document Vectors

This script demonstrates how to:
1. Generate document vectors using Sentence Transformers
2. Apply DBSCAN clustering to find natural document groupings
3. Analyze and visualize the resulting clusters
4. Store the clustering results for future use

Advantages of DBSCAN for document clustering:
- Does not require specifying the number of clusters in advance
- Can find clusters of arbitrary shape
- Can identify outliers/noise points (documents that don't fit well in any cluster)
- Well-suited for finding natural groupings in text data
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

# For vector embeddings
from sentence_transformers import SentenceTransformer

# For clustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# MongoDB connection (assuming existing setup)
from pymongo import MongoClient
from bson import ObjectId

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# MongoDB connection string
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "translation_corpus")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collection for documents and clusters
documents_collection = db["documents"]
clusters_collection = db["document_clusters"]
membership_collection = db["cluster_memberships"]

# Model for multilingual embeddings
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_document_content(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve documents from MongoDB
    
    Args:
        limit: Maximum number of documents to retrieve
        
    Returns:
        List of dictionaries with document ID and content
    """
    documents = []
    cursor = documents_collection.find(
        {"content_text": {"$exists": True, "$ne": ""}},
        {"_id": 1, "title": 1, "content_text": 1, "language_family": 1}
    ).limit(limit)
    
    for doc in cursor:
        documents.append({
            "id": str(doc["_id"]),
            "title": doc.get("title", ""),
            "content": doc.get("content_text", ""),
            "language": doc.get("language_family", "")
        })
    
    print(f"Retrieved {len(documents)} documents")
    return documents

def generate_document_vectors(documents: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Generate vector embeddings for documents using Sentence Transformers
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Dictionary mapping document IDs to vectors
    """
    # Load model
    model = SentenceTransformer(MODEL_NAME)
    print(f"Loaded model: {MODEL_NAME}")
    
    # Generate embeddings
    contents = [doc["content"] for doc in documents]
    embeddings = model.encode(contents)
    
    # Map to document IDs
    document_vectors = {}
    for i, doc in enumerate(documents):
        document_vectors[doc["id"]] = embeddings[i]
    
    print(f"Generated vectors with {embeddings.shape[1]} dimensions")
    return document_vectors

def find_optimal_epsilon(vectors: Dict[str, np.ndarray], min_samples: int = 3) -> float:
    """
    Find optimal epsilon parameter for DBSCAN using k-distance graph
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        min_samples: Minimum samples parameter for DBSCAN
        
    Returns:
        Recommended epsilon value
    """
    vector_array = np.array(list(vectors.values()))
    
    # Calculate distances to the min_samples-th nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(vector_array)
    distances, indices = nbrs.kneighbors(vector_array)
    
    # Sort distances to find the "elbow"
    k_distances = distances[:, min_samples-1]
    k_distances.sort()
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {min_samples}th nearest neighbor')
    plt.title('K-distance Graph for Optimal Epsilon Selection')
    
    # Calculate approximate "elbow" point
    # This is a simple heuristic - manual inspection of the plot is recommended
    x = np.arange(len(k_distances))
    kneedle_idx = np.argmax(k_distances[1:] - k_distances[:-1]) + 1
    optimal_epsilon = k_distances[kneedle_idx]
    
    plt.axhline(y=optimal_epsilon, color='r', linestyle='--')
    plt.text(len(k_distances)*0.5, optimal_epsilon*1.1, f'Suggested epsilon: {optimal_epsilon:.4f}')
    
    # Save plot
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/dbscan_epsilon_{timestamp}.png")
    
    print(f"Suggested optimal epsilon: {optimal_epsilon:.4f}")
    print(f"Saved epsilon analysis to {output_dir}/dbscan_epsilon_{timestamp}.png")
    
    return optimal_epsilon

def apply_dbscan_clustering(vectors: Dict[str, np.ndarray], epsilon: float = 0.5, min_samples: int = 3) -> Dict[str, Any]:
    """
    Apply DBSCAN clustering to document vectors
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        epsilon: DBSCAN epsilon parameter (maximum distance between points in a cluster)
        min_samples: DBSCAN min_samples parameter (minimum points to form a dense region)
        
    Returns:
        Dictionary with clustering results
    """
    # Convert to numpy array
    vector_array = np.array(list(vectors.values()))
    document_ids = list(vectors.keys())
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(vector_array)
    
    # Count number of clusters and noise points
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)
    
    print(f"DBSCAN found {num_clusters} clusters and {num_noise} noise points")
    print(f"Cluster distribution: {Counter(cluster_labels)}")
    
    # Calculate silhouette score (ignoring noise points)
    non_noise_indices = cluster_labels != -1
    if len(set(cluster_labels[non_noise_indices])) > 1 and sum(non_noise_indices) > 1:
        silhouette_avg = silhouette_score(
            vector_array[non_noise_indices], 
            cluster_labels[non_noise_indices]
        )
        print(f"Silhouette Score (excluding noise): {silhouette_avg:.4f}")
    else:
        silhouette_avg = 0
        print("Unable to calculate silhouette score (need at least 2 valid clusters)")
    
    # Prepare results
    results = {
        "algorithm": "dbscan",
        "parameters": {
            "epsilon": epsilon,
            "min_samples": min_samples
        },
        "num_clusters": num_clusters,
        "num_noise": num_noise,
        "silhouette_score": silhouette_avg,
        "clusters": {},
        "document_clusters": {}
    }
    
    # Document assignments and cluster information
    centroids = {}
    for cluster_id in set(cluster_labels):
        if cluster_id != -1:  # Skip noise points for centroid calculation
            # Calculate centroid as average of points in cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_vectors = vector_array[cluster_indices]
            centroid = np.mean(cluster_vectors, axis=0)
            centroids[cluster_id] = centroid
    
    # Process each document
    for i, doc_id in enumerate(document_ids):
        cluster_id = int(cluster_labels[i])
        
        # Handle noise points
        if cluster_id == -1:
            results["document_clusters"][doc_id] = {
                "cluster_id": "noise",
                "is_noise": True,
                "membership_score": 0.0
            }
            continue
        
        # Add document to appropriate cluster
        if cluster_id not in results["clusters"]:
            results["clusters"][cluster_id] = []
        
        # Calculate distance to centroid
        distance = np.linalg.norm(vector_array[i] - centroids[cluster_id])
        
        # Calculate core point score (number of neighbors within epsilon)
        neighbors = np.sum(
            np.linalg.norm(vector_array - vector_array[i], axis=1) < epsilon
        )
        core_score = neighbors / min_samples
        
        results["clusters"][cluster_id].append({
            "document_id": doc_id,
            "distance_to_centroid": float(distance),
            "core_score": float(core_score),
            "is_core_point": neighbors >= min_samples
        })
        
        # Store cluster assignment for document
        results["document_clusters"][doc_id] = {
            "cluster_id": cluster_id,
            "distance_to_centroid": float(distance),
            "core_score": float(core_score),
            "is_core_point": neighbors >= min_samples,
            "is_noise": False,
            "membership_score": 1.0  # DBSCAN gives hard assignments
        }
    
    # Sort documents within each cluster by distance to centroid
    for cluster_id in results["clusters"]:
        results["clusters"][cluster_id] = sorted(
            results["clusters"][cluster_id],
            key=lambda x: x["distance_to_centroid"]
        )
    
    # Add centroids to results
    results["centroids"] = {str(k): v.tolist() for k, v in centroids.items()}
    
    return results

def visualize_dbscan_clusters(vectors: Dict[str, np.ndarray], cluster_results: Dict[str, Any]) -> None:
    """
    Visualize DBSCAN clusters using PCA or t-SNE
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        cluster_results: Results from DBSCAN clustering
    """
    # Convert to numpy array
    vector_array = np.array(list(vectors.values()))
    document_ids = list(vectors.keys())
    
    # Get cluster assignments
    clusters = [
        cluster_results["document_clusters"][doc_id]["cluster_id"]
        for doc_id in document_ids
    ]
    
    # Convert "noise" to -1 for consistent coloring
    clusters = [-1 if c == "noise" else c for c in clusters]
    
    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vector_array)
    
    # Plot
    plt.figure(figsize=(10, 8))
    # Use distinct colors for clusters, with noise as black X
    scatter = plt.scatter(
        reduced_vectors[:, 0], 
        reduced_vectors[:, 1], 
        c=clusters, 
        cmap='viridis', 
        alpha=0.7,
        s=80
    )
    
    # Mark noise points differently
    noise_indices = [i for i, c in enumerate(clusters) if c == -1]
    if noise_indices:
        plt.scatter(
            reduced_vectors[noise_indices, 0],
            reduced_vectors[noise_indices, 1],
            marker='x',
            c='black',
            s=50,
            label='Noise'
        )
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('Document Clusters (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/dbscan_clusters_{timestamp}.png")
    print(f"Saved visualization to {output_dir}/dbscan_clusters_{timestamp}.png")
    
    # Try t-SNE for potentially better visualization
    try:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_vectors = tsne.fit_transform(vector_array)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            tsne_vectors[:, 0], 
            tsne_vectors[:, 1], 
            c=clusters, 
            cmap='viridis', 
            alpha=0.7,
            s=80
        )
        
        # Mark noise points differently
        if noise_indices:
            plt.scatter(
                tsne_vectors[noise_indices, 0],
                tsne_vectors[noise_indices, 1],
                marker='x',
                c='black',
                s=50,
                label='Noise'
            )
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('Document Clusters (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dbscan_tsne_{timestamp}.png")
        print(f"Saved t-SNE visualization to {output_dir}/dbscan_tsne_{timestamp}.png")
    except Exception as e:
        print(f"t-SNE visualization failed: {e}")

def store_dbscan_clusters_in_mongodb(cluster_results: Dict[str, Any]) -> None:
    """
    Store DBSCAN clustering results in MongoDB
    
    Args:
        cluster_results: Results from DBSCAN clustering
    """
    # Timestamp for this clustering run
    timestamp = datetime.now()
    
    # Create cluster metadata
    cluster_metadata = {
        "algorithm": "dbscan",
        "parameters": {
            "epsilon": cluster_results["parameters"]["epsilon"],
            "min_samples": cluster_results["parameters"]["min_samples"]
        },
        "quality_metrics": {
            "silhouette_score": cluster_results["silhouette_score"],
            "num_clusters": cluster_results["num_clusters"],
            "num_noise": cluster_results["num_noise"]
        },
        "feature_type": "sentence_transformer_embeddings",
        "model_name": MODEL_NAME,
        "run_date": timestamp
    }
    
    # Store cluster metadata and get ID
    metadata_id = clusters_collection.insert_one(cluster_metadata).inserted_id
    print(f"Stored cluster metadata with ID: {metadata_id}")
    
    # Store each cluster
    for cluster_id, documents in cluster_results["clusters"].items():
        # Create cluster record
        cluster_record = {
            "cluster_id": f"dbscan_{cluster_id}",
            "name": f"Cluster {cluster_id}",
            "size": len(documents),
            "centroid": cluster_results["centroids"].get(str(cluster_id)),
            "created_at": timestamp,
            "cluster_metadata_id": metadata_id
        }
        
        # Insert cluster
        cluster_doc_id = clusters_collection.insert_one(cluster_record).inserted_id
        print(f"Stored cluster {cluster_id} with ID: {cluster_doc_id}")
        
        # Store memberships
        memberships = []
        for rank, doc in enumerate(documents):
            membership = {
                "document_id": ObjectId(doc["document_id"]),
                "cluster_id": f"dbscan_{cluster_id}",
                "membership_score": 1.0,  # DBSCAN has hard assignments
                "distance_to_centroid": doc["distance_to_centroid"],
                "is_core_point": doc["is_core_point"],
                "core_score": doc["core_score"],
                "rank_in_cluster": rank + 1,
                "assigned_at": timestamp
            }
            memberships.append(membership)
        
        if memberships:
            membership_collection.insert_many(memberships)
            print(f"Stored {len(memberships)} memberships for cluster {cluster_id}")
    
    # Store noise points (if any)
    noise_docs = [
        doc_id for doc_id, info in cluster_results["document_clusters"].items()
        if info.get("is_noise", False)
    ]
    
    if noise_docs:
        # Create noise "cluster"
        noise_record = {
            "cluster_id": "dbscan_noise",
            "name": "Noise Points",
            "size": len(noise_docs),
            "centroid": None,  # No centroid for noise
            "created_at": timestamp,
            "cluster_metadata_id": metadata_id
        }
        
        noise_cluster_id = clusters_collection.insert_one(noise_record).inserted_id
        print(f"Stored noise cluster with ID: {noise_cluster_id}")
        
        # Store noise memberships
        noise_memberships = []
        for doc_id in noise_docs:
            noise_memberships.append({
                "document_id": ObjectId(doc_id),
                "cluster_id": "dbscan_noise",
                "membership_score": 0.0,
                "is_noise": True,
                "assigned_at": timestamp
            })
        
        if noise_memberships:
            membership_collection.insert_many(noise_memberships)
            print(f"Stored {len(noise_memberships)} noise point memberships")
    
def run_dbscan_experiment():
    """Run the complete DBSCAN clustering experiment"""
    
    print("Starting DBSCAN clustering experiment")
    print("======================================")
    
    # Step 1: Get document content
    documents = get_document_content(limit=200)
    
    # Step 2: Generate document vectors
    vectors = generate_document_vectors(documents)
    
    # Step 3: Find optimal epsilon parameter
    optimal_epsilon = find_optimal_epsilon(vectors, min_samples=5)
    
    # Step 4: Apply DBSCAN with optimal parameters
    cluster_results = apply_dbscan_clustering(
        vectors, 
        epsilon=optimal_epsilon,
        min_samples=5
    )
    
    # Step 5: Visualize clusters
    visualize_dbscan_clusters(vectors, cluster_results)
    
    # Step 6: Store results in MongoDB
    store_dbscan_clusters_in_mongodb(cluster_results)
    
    # Step 7: Save results to file
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"{output_dir}/dbscan_results_{timestamp}.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = json.loads(
            json.dumps(cluster_results, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
        )
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved results to {output_dir}/dbscan_results_{timestamp}.json")
    print("DBSCAN clustering experiment completed")

if __name__ == "__main__":
    run_dbscan_experiment()

    # Update document records with cluster assignments
    for doc_id, info in cluster_results["document_clusters"].items():
        # Determine cluster ID string
        if info.get("is_noise", False):
            cluster_id_str = "dbscan_noise"
        else:
            cluster_id_str = f"dbscan_{info['cluster_id']}"
        
        documents_collection.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set": {
                    "primary_cluster_id": cluster_id_str,
                    "clustering_info": {
                        "dbscan": {
                            "cluster_id": cluster_id_str,
                            "distance_to_centroid": info.get("distance_to_centroid"),
                            "is_core_point": info.get("is_core_point", False),
                            "is_noise": info.get("is_noise", False),
                            "core_score": info.get("core_score"),
                            "clustering_run_id": str(metadata_id)
                        }
                    },
                    "last_clustered_at": timestamp
                }
            }
        )
    
    print(f"Updated {len(cluster_results['document_clusters'])} documents with cluster assignments")