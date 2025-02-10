import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine  # (only used in the original, not needed now)

# =============================================================================
# PyTorch implementations for KMeans and silhouette score
# =============================================================================

def torch_kmeans(data, num_clusters, num_iters=100, tol=1e-4, device=None):
    """
    Simple k-means clustering implemented with PyTorch.
    
    Args:
        data (Tensor): shape (n_samples, n_features) – assumed to be normalized.
        num_clusters (int): number of clusters.
        num_iters (int): maximum number of iterations.
        tol (float): tolerance for centroid change.
        device: device on which to run (if None, auto-detects GPU).
    
    Returns:
        dict: A dictionary with keys:
            - "labels": Tensor of cluster assignments (n_samples,)
            - "centroids": Tensor of cluster centroids (num_clusters, n_features)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    n_samples, n_features = data.shape

    # For reproducibility, set manual seed (mimicking random_state=42)
    torch.manual_seed(42)
    # Initialize centroids by selecting random data points (without replacement)
    indices = torch.randperm(n_samples)[:num_clusters]
    centroids = data[indices].clone()

    for i in range(num_iters):
        # Compute squared Euclidean distances (using torch.cdist)
        distances = torch.cdist(data, centroids, p=2)
        # Assign each sample to the closest centroid
        labels = torch.argmin(distances, dim=1)

        new_centroids = []
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.sum() == 0:
                # If a cluster lost all its points, reinitialize it to a random point
                new_centroid = data[torch.randint(0, n_samples, (1,))]
            else:
                new_centroid = data[mask].mean(dim=0)
            new_centroids.append(new_centroid)
        new_centroids = torch.stack(new_centroids)

        # Check for convergence
        if torch.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break

        centroids = new_centroids

    return {"labels": labels.cpu(), "centroids": centroids.cpu()}


def silhouette_score_torch(data, labels):
    """
    Computes the silhouette score using PyTorch operations.
    For each sample i, the silhouette value is:
    
         s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    where:
         a(i) = mean distance of i to all other points in the same cluster,
         b(i) = minimum over other clusters of the mean distance of i to those points.
    
    Args:
        data (Tensor): shape (n_samples, n_features)
        labels (Tensor): shape (n_samples,)
    
    Returns:
        float: The average silhouette score.
    """
    n_samples = data.shape[0]
    if n_samples <= 1:
        return 0.0

    # Compute full pairwise distance matrix (Euclidean distances)
    dist_matrix = torch.cdist(data, data, p=2)
    sil_scores = []

    unique_labels = labels.unique()
    for i in range(n_samples):
        # Boolean mask for points in the same cluster as i
        same_cluster = (labels == labels[i])
        # Exclude self by temporarily setting its mask to False
        same_cluster_indices = same_cluster.nonzero(as_tuple=True)[0]
        if same_cluster_indices.numel() > 1:
            # a(i): mean distance to all other points in the same cluster
            a = (dist_matrix[i, same_cluster_indices].sum() - 0.0) / (same_cluster_indices.numel() - 1)
        else:
            a = 0.0

        # For b(i): compute mean distance to each other cluster and take the minimum
        b = float('inf')
        for cl in unique_labels:
            if cl == labels[i]:
                continue
            other_indices = (labels == cl).nonzero(as_tuple=True)[0]
            if other_indices.numel() > 0:
                b_candidate = dist_matrix[i, other_indices].mean().item()
                if b_candidate < b:
                    b = b_candidate
        # Compute the silhouette score for sample i
        if max(a, b) > 0:
            s = (b - a) / max(a, b)
        else:
            s = 0.0
        sil_scores.append(s)
    return sum(sil_scores) / len(sil_scores)


def binary_search_optimal_kmeans(data, min_k, max_k):
    """
    Finds the optimal k (number of clusters) using binary search over the silhouette score.
    
    Args:
        data (Tensor): Normalized data tensor.
        min_k (int): Minimum k to try.
        max_k (int): Maximum k to try.
    
    Returns:
        dict: A dictionary containing:
            - "labels": Cluster assignments.
            - "centroids": Cluster centers.
    """
    best_score = -1
    # Start with 1 cluster (if the data is too small)
    best_kmeans = torch_kmeans(data, 1)
    
    # We only try k>=2 because silhouette score is undefined for a single cluster.
    while min_k <= max_k:
        mid_k = (min_k + max_k) // 2
        if mid_k < 2:
            break
        kmeans_result = torch_kmeans(data, mid_k)
        labels = kmeans_result["labels"].to(data.device)
        score = silhouette_score_torch(data, labels)
        if score > best_score:
            best_score = score
            best_kmeans = kmeans_result  # update best clustering
            min_k = mid_k + 1
        else:
            max_k = mid_k - 1
    return best_kmeans

# =============================================================================
# Main functions (updated to use PyTorch operators)
# =============================================================================

def compute_kmeans(sentences):
    """
    Computes K-means clustering for a list of sentences by generating their embeddings,
    normalizing them, and determining the optimal number of clusters using binary search.
    
    Args:
        sentences (list): List of sentences to be clustered.
    
    Returns:
        tuple: (embeddings, kmeans_result) where:
            - embeddings (Tensor): Normalized sentence embeddings.
            - kmeans_result (dict): Dictionary with keys "labels" and "centroids".
    """
    # Use SentenceTransformer with convert_to_tensor=True so that embeddings are a torch.Tensor
    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    embeddings = model.encode(sentences, convert_to_tensor=True)
    # Normalize embeddings (L2 normalization along feature dimension)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Determine optimal clustering with binary search (using number of clusters from 1 to len(sentences))
    # (Note: binary search is only meaningful when len(sentences)>=2)
    n = embeddings.shape[0]
    kmeans_result = binary_search_optimal_kmeans(embeddings, min_k=1, max_k=(n - 1) if n > 1 else 1)
    return embeddings, kmeans_result


def compute_largest_cluster(sentences):
    """
    Computes the largest cluster of sentences using k-means clustering,
    then returns the embeddings and the sentences from that cluster ordered
    by their proximity (cosine distance) to the cluster center.
    
    Args:
        sentences (list): List of sentences.
    
    Returns:
        tuple: (embeddings, sentences_of_largest_cluster)
            - embeddings (Tensor): Normalized embeddings.
            - sentences_of_largest_cluster (list): Sentences in the largest cluster, sorted.
    """
    if len(sentences) == 0:
        return None, None
    
    embeddings, kmeans_result = compute_kmeans(sentences)
    labels = kmeans_result["labels"]
    
    # Compute cluster sizes using torch.bincount
    cluster_sizes = torch.bincount(labels)
    largest_cluster_idx = torch.argmax(cluster_sizes)
    
    # Get indices of sentences belonging to the largest cluster
    cluster_member_ids = torch.where(labels == largest_cluster_idx)[0]
    # Extract corresponding sentences (as a list)
    sentences_of_largest_cluster = [sentences[i] for i in cluster_member_ids.tolist()]
    
    # Get the centroid (cluster center) of the largest cluster
    largest_cluster_center = kmeans_result["centroids"][largest_cluster_idx]
    # Get the embeddings of the largest cluster members
    embeddings_largest = embeddings[cluster_member_ids]
    
    # Compute cosine distances:
    # Since the embeddings are normalized, cosine similarity is just the dot product.
    # Cosine distance = 1 - cosine similarity.
    # Expand the centroid to match the number of cluster members.
    centroid_expanded = largest_cluster_center.unsqueeze(0).expand(embeddings_largest.size(0), -1)
    cosine_similarities = F.cosine_similarity(embeddings_largest, centroid_expanded, dim=1)
    cosine_distances = 1 - cosine_similarities  # lower distance means more central
    
    # Sort the sentences by increasing cosine distance
    sorted_indices = torch.argsort(cosine_distances)
    sentences_of_largest_cluster = [sentences_of_largest_cluster[i] for i in sorted_indices.tolist()]
    
    return embeddings, sentences_of_largest_cluster


def flatten_values_lists_of_list_dicts_to_dict(item):
    """
    Flattens a list of dictionaries containing lists of values into a single dictionary.
    
    Args:
        item (list): List of dictionaries, where each dictionary's values are lists.
    
    Returns:
        dict: A dictionary where each key has a flattened list of values.
    """
    result = {}
    for i in item:
        if isinstance(i, list):
            i = i[0]
        for key, lists in i.items():
            if key not in result:
                result[key] = []
            result[key].extend(lists)
    return result


def gather_processes(local_candidates, local_references=None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("RANK", "0"))
    global_candidates_list = None
    global_references_list = None

    if local_rank == 0:
        # Initialize the gather list only on the root process
        global_candidates_list = [None for _ in range(world_size)]
        global_references_list = [None for _ in range(world_size)]
    try:
        dist.gather_object(local_candidates, global_candidates_list, dst=0)

        if local_references is not None:
            dist.gather_object(local_references, global_references_list, dst=0)

    except Exception as e:
        print(f"Error during result gathering: {e}")

    if local_rank != 0:
        # Clean up and exit if not root.
        dist.destroy_process_group()
        sys.exit()

    # Flatten the gathered list of candidates
    candidates_list = []
    for i in global_candidates_list:
        candidates_list.extend(i)

    if global_references_list[0] is not None:
        references_list = []
        for i in global_references_list:
            references_list.extend(i)
        print(f"References list: {len(references_list)}")
        return candidates_list, references_list

    return candidates_list


def clean_responses(response):
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if ("[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>") in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def make_prompt(text1, text2, max_len=300):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in
    comparison to a reference radiology report.
    
    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.
    
    Returns:
        str: Formatted prompt.
    """
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = (
        "Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n"
        "    Process Overview: You will be presented with:\n\n"
        "    1. The criteria for making a judgment.\n"
        "    2. The reference radiology report.\n"
        "    3. The candidate radiology report.\n"
        "    4. The desired format for your assessment.\n\n"
        "    1. Criteria for Judgment:\n\n"
        "    For each candidate report, determine:\n\n"
        "    The count of clinically significant errors.\n"
        "    The count of clinically insignificant errors.\n\n"
        "    Errors can fall into one of these categories:\n\n"
        "    a) False report of a finding in the candidate.\n"
        "    b) Missing a finding present in the reference.\n"
        "    c) Misidentification of a finding's anatomic location/position.\n"
        "    d) Misassessment of the severity of a finding.\n"
        "    e) Mentioning a comparison that isn't in the reference.\n"
        "    f) Omitting a comparison detailing a change from a prior study.\n"
        "    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n"
        "    2. Reference Report:\n    " + text1 + "\n\n"
        "    3. Candidate Report:\n    " + text2 + "\n\n"
        "    4. Reporting Your Assessment:\n\n"
        "    Follow this specific format for your output, even if no errors are found:\n"
        "    ```\n"
        "    [Explanation]:\n"
        "    <Explanation>\n\n"
        "    [Clinically Significant Errors]:\n"
        "    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n"
        "    ....\n"
        "    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n"
        "    [Clinically Insignificant Errors]:\n"
        "    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n"
        "    ....\n"
        "    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n"
        "    [Matched Findings]:\n"
        "    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n"
        "    ```\n"
    )
    return prompt