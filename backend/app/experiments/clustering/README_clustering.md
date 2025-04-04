# Document Clustering Experiments

This directory contains experiments for document clustering in the corpus-graph-document-translation project.

## Overview

These scripts implement various clustering algorithms to group similar documents in the translation corpus.
The goal is to discover natural groupings of documents that share similar content, terminology, or translation patterns.

## Available Clustering Approaches

### 1. K-Means Clustering (`kmeans_clustering.py`)

K-Means divides documents into k clusters where each document belongs to the cluster with the nearest mean.

- **Pros**: Modern embedding quality with simple clustering
- **Cons**: Requires specifying k (number of clusters)

**Best for**:
- When you have an idea of how many clusters you want
- When you expect clusters to be roughly similar in size
- Finding general topic groupings
- Document retrieval and finding similar translation examples

**Features**:
- Automatically finds optimal k using elbow method and silhouette scores
- Visualizes clusters using PCA and t-SNE 
- Generates cluster centroids for efficient similarity lookups

### 2. DBSCAN Clustering (`dbscan_clustering.py`)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are close to each other in the feature space and marks points in low-density regions as outliers.

- **Pros**: Doesn't require pre-specifying number of clusters, good at finding natural groupings
- **Cons**: Requires careful parameter tuning

**Best for**:
- Discovering natural document groupings without specifying the number
- Detecting outlier documents that don't fit well with others
- Finding clusters of varying sizes and shapes

**Features**:
- Automatically determines appropriate epsilon parameter
- Identifies core documents and peripheral documents in each cluster
- Recognizes "noise" documents that don't belong to any cluster

### 3. Hierarchical Clustering (`hierarchical_clustering.py`)

Hierarchical clustering builds a tree of clusters, providing multiple levels of document grouping.

- **Pros**: Creates a tree of document similarities, works well for domain categorization
- **Cons**: Computationally expensive for large document sets

**Best for**:
- Creating taxonomies of documents
- Exploring document relationships at different granularity levels
- Understanding the hierarchical relationship between documents (e.g., broad domains → specific topics)
- When you want both broad categories and specific subcategories

**Features**:
- Creates dendrograms showing document relationships
- Provides multiple clustering levels based on distance thresholds
- Supports exploration of cluster hierarchies

### 4. Topic Modeling (LDA)
- **Pros**: Discovers underlying topics and their distribution across documents
- **Cons**: Topics are distributions over words rather than strict clusters
- **Best for**: Understanding thematic content that can inform translation

## Usage

### Prerequisites

1. Make sure you have the required Python packages installed:
   ```
   pip install -r requirements.txt
   ```

2. Configure MongoDB connection:
   Create a `.env` file with your MongoDB connection details:
   ```
   MONGODB_URI=mongodb://localhost:27017
   DB_NAME=Auto-Translator
   ```

3. Update the collection names in the scripts if your database uses different collection names than the default ones in the scripts. Look for lines in the scripts that define the collections:

    ```
    documents_collection = db["documents"]
    clusters_collection = db["document_clusters"]
    membership_collection = db["cluster_memberships"]
    ```

### Running Experiments
**Recommended Approach for Your Project**:
Based on your project's focus on translation and multilingual corpora:

- **Start with DBSCAN** - It works well with multilingual document content and doesn't require specifying the number of clusters in advance.
- **Try hierarchical clustering next** - This will give you a rich view of how documents relate at different levels (e.g., broad domain → specific topic → terminology), which is valuable for translation contexts.
- **Use K-means as a baseline** - It's a well-understood algorithm that can serve as a good comparison point.

1. To run DBSCAN clustering:
   ```
   python dbscan_clustering.py
   ```

2. To run Hierarchical clustering:
   ```
   python hierarchical_clustering.py
   ```

3. To run K-Means clustering:
   ```
   python kmeans_clustering.py
   ```

### Experiment Results

All experiments save their results to the `clustering_results` directory in MongoDB so you can compare approaches before deciding which one to integrate into your main codebase:
- **Cluster visualizations**: PNG image files
- **Cluster data**: JSON files with detailed information
- **Parameter analysis**: Images showing optimal parameter selection

## Integrating with Main Application

Once you've determined which clustering approach works best for your corpus, you can integrate
it into the main application by:

1. Add the clustering models from `clustering_models.py` to your main `models.py` file
2. Update your DocumentMetadata class to include clustering information
3. Add database collections for clusters and memberships
4. Create a service component that runs clustering periodically or on-demand

## Data Model

The clustering experiments use the following data model:

- **ClusterMetadata**: Information about how clusters were generated
- **DocumentCluster**: A cluster of similar documents
- **ClusterMembership**: A document's membership in a cluster
- **DocumentVector**: Vector representation of a document

These models can be found in `clustering_models.py`.

## Tips for Choosing the Right Algorithm

- **K-Means**: Use when document groupings are expected to be roughly equal in size and you have a sense of the number of topics
- **DBSCAN**: Use when you want to discover natural groupings and identify outliers
- **Hierarchical**: Use when you want to explore relationships at multiple levels of granularity

For translation corpora specifically, hierarchical clustering often works well because document relationships
can exist at multiple levels (domain, subject area, terminology usage, etc.).