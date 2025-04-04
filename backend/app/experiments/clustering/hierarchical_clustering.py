"""
Hierarchical Clustering Test for Document Vectors

This script demonstrates how to:
1. Generate document vectors using Sentence Transformers
2. Apply hierarchical clustering to organize documents in a tree structure
3. Visualize the hierarchy using dendrograms
4. Store the hierarchical clustering results for future use

Advantages of hierarchical clustering for document organization:
- Creates a tree-like structure of document relationships
- Allows multi-level exploration of document similarities
- Useful for organizing documents in logical hierarchies
- Can inspect clusters at different levels of granularity
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# For vector embeddings
from sentence_transformers import SentenceTransformer

# For clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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

def apply_hierarchical_clustering(
    vectors: Dict[str, np.ndarray], 
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    linkage_method: str = 'ward'
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Apply hierarchical clustering to document vectors
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        n_clusters: Number of clusters to create (if None, distance_threshold is used)
        distance_threshold: Maximum distance to form clusters (if None, n_clusters is used)
        linkage_method: Method for calculating distances between clusters
        
    Returns:
        Dictionary with clustering results and linkage matrix
    """
    # Convert to numpy array
    vector_array = np.array(list(vectors.values()))
    document_ids = list(vectors.keys())
    
    # Calculate linkage matrix
    linkage_matrix = linkage(vector_array, method=linkage_method)
    
    # Apply hierarchical clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage_method
    )
    cluster_labels = hierarchical.fit_predict(vector_array)
    
    # Number of clusters
    num_clusters = len(set(cluster_labels))
    print(f"Hierarchical clustering found {num_clusters} clusters")
    
    # Calculate silhouette score
    if num_clusters > 1:
        silhouette_avg = silhouette_score(vector_array, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
    else:
        silhouette_avg = 0
        print("Only one cluster found, silhouette score not applicable")
    
    # Prepare results
    results = {
        "algorithm": "hierarchical",
        "parameters": {
            "n_clusters": n_clusters,
            "distance_threshold": distance_threshold,
            "linkage_method": linkage_method
        },
        "num_clusters": num_clusters,
        "silhouette_score": silhouette_avg,
        "clusters": defaultdict(list),
        "document_clusters": {},
        "levels": {}  # Store different clustering levels
    }
    
    # Document assignments
    for i, doc_id in enumerate(document_ids):
        cluster_id = int(cluster_labels[i])
        results["clusters"][cluster_id].append({
            "document_id": doc_id
        })
        
        # Store cluster assignment for document
        results["document_clusters"][doc_id] = {
            "cluster_id": cluster_id
        }
    
    # Extract multi-level clustering (for different distance thresholds)
    max_distance = linkage_matrix[-1, 2]
    
    # Choose a set of meaningful distance thresholds
    distance_thresholds = [
        max_distance * factor for factor in [0.3, 0.5, 0.7]
    ]
    
    # Get clusters at different levels
    for level, thresh in enumerate(distance_thresholds):
        level_labels = fcluster(linkage_matrix, thresh, criterion='distance')
        level_clusters = defaultdict(list)
        
        for i, doc_id in enumerate(document_ids):
            level_cluster_id = int(level_labels[i])
            level_clusters[level_cluster_id].append(doc_id)
            
            # Add to document info
            if "levels" not in results["document_clusters"][doc_id]:
                results["document_clusters"][doc_id]["levels"] = {}
            
            results["document_clusters"][doc_id]["levels"][f"level_{level}"] = {
                "cluster_id": level_cluster_id,
                "distance_threshold": thresh
            }
        
        # Store level clusters
        results["levels"][f"level_{level}"] = {
            "distance_threshold": thresh,
            "num_clusters": len(level_clusters),
            "clusters": dict(level_clusters)
        }
    
    return results, linkage_matrix

def visualize_hierarchical_clusters(
    vectors: Dict[str, np.ndarray], 
    cluster_results: Dict[str, Any],
    linkage_matrix: np.ndarray,
    document_titles: Optional[Dict[str, str]] = None
) -> None:
    """
    Visualize hierarchical clusters using dendrograms and PCA
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        cluster_results: Results from hierarchical clustering
        linkage_matrix: Linkage matrix from hierarchical clustering
        document_titles: Optional dictionary mapping document IDs to titles
    """
    # Convert to numpy array
    vector_array = np.array(list(vectors.values()))
    document_ids = list(vectors.keys())
    
    # Create output directory
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Plot dendrogram
    plt.figure(figsize=(12, 8))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Document Index')
    plt.ylabel('Distance')
    
    # Simplify dendrogram if many documents
    if len(document_ids) > 50:
        # Truncate dendrogram for readability
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=30,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True
        )
    else:
        # Full dendrogram with document labels
        labels = [
            document_titles.get(doc_id, f"Doc {i}") if document_titles 
            else f"Doc {i}" 
            for i, doc_id in enumerate(document_ids)
        ]
        dendrogram(
            linkage_matrix,
            orientation='top',
            labels=labels,
            leaf_rotation=90.,
            leaf_font_size=10.
        )
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hierarchical_dendrogram_{timestamp}.png")
    print(f"Saved dendrogram to {output_dir}/hierarchical_dendrogram_{timestamp}.png")
    
    # 2. Plot clusters with PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vector_array)
    
    # Get cluster assignments
    clusters = [
        cluster_results["document_clusters"][doc_id]["cluster_id"]
        for doc_id in document_ids
    ]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_vectors[:, 0], 
        reduced_vectors[:, 1], 
        c=clusters, 
        cmap='viridis', 
        alpha=0.7
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title('Document Clusters (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hierarchical_clusters_{timestamp}.png")
    print(f"Saved cluster visualization to {output_dir}/hierarchical_clusters_{timestamp}.png")
    
    # 3. Plot multiple levels if available
    if cluster_results.get("levels"):
        level_names = sorted(cluster_results["levels"].keys())
        
        # Select up to 3 levels for visualization
        visualization_levels = level_names[:min(3, len(level_names))]
        
        # Plot each level
        for level_name in visualization_levels:
            level_info = cluster_results["levels"][level_name]
            
            # Get cluster assignments for this level
            level_clusters = []
            for doc_id in document_ids:
                level_cluster = cluster_results["document_clusters"][doc_id].get("levels", {}).get(
                    level_name, {}).get("cluster_id", 0)
                level_clusters.append(level_cluster)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                reduced_vectors[:, 0], 
                reduced_vectors[:, 1], 
                c=level_clusters, 
                cmap='viridis', 
                alpha=0.7
            )
            plt.colorbar(scatter, label='Cluster')
            plt.title(f'Document Clusters - {level_name} (Threshold: {level_info["distance_threshold"]:.2f})')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/hierarchical_clusters_{level_name}_{timestamp}.png")
            print(f"Saved {level_name} visualization to {output_dir}/hierarchical_clusters_{level_name}_{timestamp}.png")

def store_hierarchical_clusters_in_mongodb(cluster_results: Dict[str, Any]) -> None:
    """
    Store hierarchical clustering results in MongoDB
    
    Args:
        cluster_results: Results from hierarchical clustering
    """
    # Timestamp for this clustering run
    timestamp = datetime.now()
    
    # Create cluster metadata
    cluster_metadata = {
        "algorithm": "hierarchical",
        "parameters": cluster_results["parameters"],
        "quality_metrics": {
            "silhouette_score": cluster_results["silhouette_score"],
            "num_clusters": cluster_results["num_clusters"]
        },
        "feature_type": "sentence_transformer_embeddings",
        "model_name": MODEL_NAME,
        "run_date": timestamp
    }
    
    # Store cluster metadata and get ID
    metadata_id = clusters_collection.insert_one(cluster_metadata).inserted_id
    print(f"Stored cluster metadata with ID: {metadata_id}")
    
    # Store top-level clusters
    for cluster_id, documents in cluster_results["clusters"].items():
        # Create cluster record
        cluster_record = {
            "cluster_id": f"hier_{cluster_id}",
            "name": f"Cluster {cluster_id}",
            "size": len(documents),
            "created_at": timestamp,
            "cluster_metadata_id": metadata_id
        }
        
        # Insert cluster
        cluster_doc_id = clusters_collection.insert_one(cluster_record).inserted_id
        print(f"Stored cluster {cluster_id} with ID: {cluster_doc_id}")
        
        # Store memberships
        memberships = []
        for doc in documents:
            membership = {
                "document_id": ObjectId(doc["document_id"]),
                "cluster_id": f"hier_{cluster_id}",
                "membership_score": 1.0,  # Hard assignments
                "assigned_at": timestamp
            }
            memberships.append(membership)
        
        if memberships:
            membership_collection.insert_many(memberships)
            print(f"Stored {len(memberships)} memberships for cluster {cluster_id}")
    
    # Store levels if available
    if cluster_results.get("levels"):
        for level_name, level_info in cluster_results["levels"].items():
            # Store level metadata
            level_metadata = {
                "level_name": level_name,
                "distance_threshold": level_info["distance_threshold"],
                "num_clusters": level_info["num_clusters"],
                "hierarchical_clustering_id": metadata_id,
                "created_at": timestamp
            }
            
            level_id = clusters_collection.insert_one(level_metadata).inserted_id
            print(f"Stored level metadata for {level_name} with ID: {level_id}")
            
            # Store level clusters
            for level_cluster_id, doc_ids in level_info["clusters"].items():
                level_cluster_record = {
                    "cluster_id": f"hier_{level_name}_{level_cluster_id}",
                    "name": f"Level {level_name} Cluster {level_cluster_id}",
                    "size": len(doc_ids),
                    "level": level_name,
                    "level_metadata_id": level_id,
                    "created_at": timestamp
                }
                
                level_cluster_doc_id = clusters_collection.insert_one(level_cluster_record).inserted_id
                
                # Store memberships for this level cluster
                level_memberships = [
                    {
                        "document_id": ObjectId(doc_id),
                        "cluster_id": f"hier_{level_name}_{level_cluster_id}",
                        "membership_score": 1.0,
                        "level": level_name,
                        "assigned_at": timestamp
                    }
                    for doc_id in doc_ids
                ]
                
                if level_memberships:
                    membership_collection.insert_many(level_memberships)
    
    # Update document records with cluster assignments
    for doc_id, info in cluster_results["document_clusters"].items():
        update_data = {
            "primary_cluster_id": f"hier_{info['cluster_id']}",
            "clustering_info": {
                "hierarchical": {
                    "cluster_id": f"hier_{info['cluster_id']}",
                    "clustering_run_id": str(metadata_id)
                }
            },
            "last_clustered_at": timestamp
        }
        
        # Add level information if available
        if info.get("levels"):
            update_data["clustering_info"]["hierarchical"]["levels"] = {}
            for level_name, level_info in info["levels"].items():
                update_data["clustering_info"]["hierarchical"]["levels"][level_name] = {
                    "cluster_id": f"hier_{level_name}_{level_info['cluster_id']}",
                    "distance_threshold": level_info["distance_threshold"]
                }
        
        documents_collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": update_data}
        )
    
    print(f"Updated {len(cluster_results['document_clusters'])} documents with cluster assignments")

def determine_optimal_clusters(
    vectors: Dict[str, np.ndarray], 
    max_clusters: int = 10,
    linkage_method: str = 'ward'
) -> int:
    """
    Determine optimal number of clusters for hierarchical clustering
    
    Args:
        vectors: Dictionary mapping document IDs to vectors
        max_clusters: Maximum number of clusters to try
        linkage_method: Method for calculating distances between clusters
        
    Returns:
        Optimal number of clusters
    """
    vector_array = np.array(list(vectors.values()))
    
    # Calculate linkage matrix once
    print("Calculating linkage matrix...")
    linkage_matrix = linkage(vector_array, method=linkage_method)
    
    # Calculate silhouette scores for different numbers of clusters
    silhouette_scores = []
    k_values = range(2, min(max_clusters + 1, len(vectors)))
    
    for k in k_values:
        print(f"Testing k={k}...")
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        
        # Calculate silhouette score
        score = silhouette_score(vector_array, labels)
        silhouette_scores.append(score)
        print(f"  Silhouette score: {score:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_values), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.grid(True)
    
    # Save plot
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{output_dir}/hierarchical_optimal_k_{timestamp}.png")
    
    # Get optimal k (maximize silhouette score)
    optimal_k = list(k_values)[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Saved optimal k analysis to {output_dir}/hierarchical_optimal_k_{timestamp}.png")
    
    return optimal_k

def run_hierarchical_experiment():
    """Run the complete hierarchical clustering experiment"""
    
    print("Starting hierarchical clustering experiment")
    print("===========================================")
    
    # Step 1: Get document content
    documents = get_document_content(limit=200)
    
    # Create a mapping of document IDs to titles for better visualization
    document_titles = {doc["id"]: doc["title"] for doc in documents}
    
    # Step 2: Generate document vectors
    vectors = generate_document_vectors(documents)
    
    # Step 3: Find optimal number of clusters
    optimal_k = determine_optimal_clusters(vectors, max_clusters=15)
    
    # Step 4: Apply hierarchical clustering with optimal clusters
    cluster_results, linkage_matrix = apply_hierarchical_clustering(
        vectors, 
        n_clusters=optimal_k,
        linkage_method='ward'
    )
    
    # Step 5: Visualize clusters
    visualize_hierarchical_clusters(vectors, cluster_results, linkage_matrix, document_titles)
    
    # Step 6: Store results in MongoDB
    store_hierarchical_clusters_in_mongodb(cluster_results)
    
    # Step 7: Save results to file
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"{output_dir}/hierarchical_results_{timestamp}.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = json.loads(
            json.dumps(cluster_results, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
        )
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved results to {output_dir}/hierarchical_results_{timestamp}.json")
    print("Hierarchical clustering experiment completed")

if __name__ == "__main__":
    run_hierarchical_experiment()