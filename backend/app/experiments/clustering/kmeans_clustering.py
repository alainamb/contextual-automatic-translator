"""
K-Means Clustering Test for Document Vectors

This script demonstrates how to:
1. Generate document vectors using Sentence Transformers
2. Apply K-means clustering to group similar documents
3. Analyze and visualize the resulting clusters
4. Store the clustering results for future use
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional

# For vector embeddings
from sentence_transformers import SentenceTransformer

# For clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def apply_kmeans_clustering(vectors: Dict[str, np.ndarray], n_clusters: int = 5) -> Dict[str, Any]:
    """
    Apply K-means clustering to document vectors
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        n_clusters: Number of clusters to create
        
    Returns:
        Dictionary with clustering results
    """
    # Convert to numpy array
    vector_array = np.array(list(vectors.values()))
    document_ids = list(vectors.keys())
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(vector_array)
    
    # Calculate silhouette score
    if len(set(cluster_labels)) > 1:  # Silhouette score needs at least 2 clusters
        silhouette_avg = silhouette_score(vector_array, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
    else:
        silhouette_avg = 0
        print("Only one cluster found, silhouette score not applicable")
    
    # Prepare results
    results = {
        "algorithm": "kmeans",
        "n_clusters": n_clusters,
        "silhouette_score": silhouette_avg,
        "clusters": {},
        "document_clusters": {},
        "centroids": kmeans.cluster_centers_.tolist()
    }
    
    # Document assignments
    for i, doc_id in enumerate(document_ids):
        cluster_id = int(cluster_labels[i])
        
        # Add document to appropriate cluster
        if cluster_id not in results["clusters"]:
            results["clusters"][cluster_id] = []
        
        # Calculate distance to centroid
        distance = np.linalg.norm(vector_array[i] - kmeans.cluster_centers_[cluster_id])
        
        results["clusters"][cluster_id].append({
            "document_id": doc_id,
            "distance": float(distance)
        })
        
        # Store cluster assignment for document
        results["document_clusters"][doc_id] = {
            "cluster_id": cluster_id,
            "distance": float(distance)
        }
    
    # Sort documents within each cluster by distance
    for cluster_id in results["clusters"]:
        results["clusters"][cluster_id] = sorted(
            results["clusters"][cluster_id],
            key=lambda x: x["distance"]
        )
    
    print(f"Created {n_clusters} clusters")
    return results

def visualize_clusters(vectors: Dict[str, np.ndarray], cluster_results: Dict[str, Any]) -> None:
    """
    Visualize clusters using PCA or t-SNE
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        cluster_results: Results from K-means clustering
    """
    # Convert to numpy array
    vector_array = np.array(list(vectors.values()))
    document_ids = list(vectors.keys())
    
    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vector_array)
    
    # Get cluster assignments
    clusters = [cluster_results["document_clusters"][doc_id]["cluster_id"] for doc_id in document_ids]
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Document Clusters (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    
    # Save plot
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/kmeans_clusters_{timestamp}.png")
    print(f"Saved visualization to {output_dir}/kmeans_clusters_{timestamp}.png")
    
    # Try t-SNE for potentially better visualization
    try:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_vectors = tsne.fit_transform(vector_array)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Document Clusters (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/kmeans_tsne_{timestamp}.png")
        print(f"Saved t-SNE visualization to {output_dir}/kmeans_tsne_{timestamp}.png")
    except Exception as e:
        print(f"t-SNE visualization failed: {e}")

def store_clusters_in_mongodb(cluster_results: Dict[str, Any]) -> None:
    """
    Store clustering results in MongoDB
    
    Args:
        cluster_results: Results from K-means clustering
    """
    # Timestamp for this clustering run
    timestamp = datetime.now()
    
    # Create cluster metadata
    cluster_metadata = {
        "algorithm": "kmeans",
        "parameters": {
            "n_clusters": cluster_results["n_clusters"]
        },
        "quality_metrics": {
            "silhouette_score": cluster_results["silhouette_score"]
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
            "cluster_id": f"kmeans_{cluster_id}",
            "name": f"Cluster {cluster_id}",
            "size": len(documents),
            "centroid": cluster_results["centroids"][cluster_id],
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
                "cluster_id": f"kmeans_{cluster_id}",
                "membership_score": 1.0,  # K-means has hard assignments
                "distance_to_centroid": doc["distance"],
                "rank_in_cluster": rank + 1,
                "assigned_at": timestamp
            }
            memberships.append(membership)
        
        if memberships:
            membership_collection.insert_many(memberships)
            print(f"Stored {len(memberships)} memberships for cluster {cluster_id}")
    
    # Update document records with cluster assignments
    for doc_id, info in cluster_results["document_clusters"].items():
        documents_collection.update_one(
            {"_id": ObjectId(doc_id)},
            {
                "$set": {
                    "primary_cluster_id": f"kmeans_{info['cluster_id']}",
                    "clustering_info": {
                        "kmeans": {
                            "cluster_id": f"kmeans_{info['cluster_id']}",
                            "distance_to_centroid": info["distance"],
                            "clustering_run_id": str(metadata_id)
                        }
                    },
                    "last_clustered_at": timestamp
                }
            }
        )
    
    print(f"Updated {len(cluster_results['document_clusters'])} documents with cluster assignments")

def determine_optimal_k(vectors: Dict[str, np.ndarray], max_k: int = 10) -> int:
    """
    Determine optimal number of clusters using the elbow method
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        max_k: Maximum number of clusters to try
        
    Returns:
        Optimal number of clusters
    """
    vector_array = np.array(list(vectors.values()))
    
    # Calculate inertia (sum of squared distances to centroids)
    inertia_values = []
    silhouette_values = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(vector_array)
        inertia_values.append(kmeans.inertia_)
        
        # Silhouette score
        if len(set(labels)) > 1:  # Need at least 2 clusters
            silhouette_avg = silhouette_score(vector_array, labels)
            silhouette_values.append(silhouette_avg)
        else:
            silhouette_values.append(0)
    
    # Plot elbow method results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia_values, 'bo-')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_values, 'ro-')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/kmeans_optimal_k_{timestamp}.png")
    print(f"Saved optimal k analysis to {output_dir}/kmeans_optimal_k_{timestamp}.png")
    
    # Find optimal k (you may need to manually interpret these plots)
    # Here's a simple method: find where silhouette score is highest
    optimal_k = k_values[np.argmax(silhouette_values)]
    print(f"Optimal k based on silhouette score: {optimal_k}")
    
    return optimal_k

def run_kmeans_experiment():
    """Run the complete K-means clustering experiment"""
    
    print("Starting K-means clustering experiment")
    print("======================================")
    
    # Step 1: Get document content
    documents = get_document_content(limit=200)
    
    # Step 2: Generate document vectors
    vectors = generate_document_vectors(documents)
    
    # Step 3: Find optimal number of clusters
    optimal_k = determine_optimal_k(vectors, max_k=10)
    
    # Step 4: Apply K-means with optimal k
    cluster_results = apply_kmeans_clustering(vectors, n_clusters=optimal_k)
    
    # Step 5: Visualize clusters
    visualize_clusters(vectors, cluster_results)
    
    # Step 6: Store results in MongoDB
    store_clusters_in_mongodb(cluster_results)
    
    # Step 7: Save results to file
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"{output_dir}/kmeans_results_{timestamp}.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = json.loads(
            json.dumps(cluster_results, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
        )
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved results to {output_dir}/kmeans_results_{timestamp}.json")
    print("K-means clustering experiment completed")

if __name__ == "__main__":
    run_kmeans_experiment()