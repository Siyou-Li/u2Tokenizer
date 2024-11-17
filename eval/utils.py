import torch.distributed as dist
import os
import sys

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import numpy as np

# A dictionary to store rewards for pairs of reference and hypothesis reports


def compute_largest_cluster(sentences):
    """
    Computes the largest cluster of sentences using K-means clustering, finds the sentences within the largest cluster, and orders them by their distance to the cluster center.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - sentences_of_largest_cluster (list): Sentences in the largest cluster, ordered by their proximity
              to the cluster center.
    """
    if len(sentences) == 0:
        return None, None
    embeddings, kmeans = compute_kmeans(sentences)
    cluster_sizes = np.bincount(kmeans.labels_)
    largest_cluster_idx = np.argmax(cluster_sizes)
    cluster_member_ids = np.where(kmeans.labels_ == largest_cluster_idx)[0]
    sentences_of_largest_cluster = [sentences[i] for i in cluster_member_ids]

    largest_cluster_mean = kmeans.cluster_centers_[largest_cluster_idx]
    embeddings_of_largest_cluster = [embeddings[i] for i in cluster_member_ids]
    distances = distance.cdist(
        embeddings_of_largest_cluster, [largest_cluster_mean], "cosine"
    ).flatten()
    closest_point_indices = np.argsort(distances)[0]

    sentences_of_largest_cluster = sentences_of_largest_cluster[closest_point_indices]

    return embeddings, sentences_of_largest_cluster


def compute_kmeans(sentences):
    """
    Computes K-means clustering for a list of sentences by generating their embeddings, normalizing the embeddings, and determining the optimal number of clusters using binary search.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - kmeans (KMeans): The KMeans object with the optimal number of clusters determined.
    """
    # sentence embeddings
    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    embeddings = model.encode(sentences)
    # normalize the embeddings for equivalent computation of the cosine distance
    embeddings = preprocessing.normalize(embeddings)
    # compute the number of clusters with binary search
    kmeans = binary_search_optimal_kmeans(embeddings, min_k=0, max_k=len(sentences))
    return embeddings, kmeans


def binary_search_optimal_kmeans(data, min_k, max_k):
    """
    Finds the optimal k for KMeans clustering using binary search on the silhouette score.

    Args:
        data (list): cluster data.
        min_k: minimum k for binary search
        max_k: maximum k for binary search

    Returns:
        list: List of cleaned response strings.
    """
    best_k = min_k
    best_score = -1
    best_kmeans = KMeans(n_clusters=1, random_state=42).fit(
        data
    )  # start with 1 cluster for len(data) < 2

    while min_k <= max_k:
        mid_k = (min_k + max_k) // 2
        if mid_k < 2:
            break

        kmeans = KMeans(n_clusters=mid_k, random_state=42).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)

        if score > best_score:
            best_score = score
            best_k = mid_k
            best_kmeans = kmeans  # Update the best KMeans model
            min_k = mid_k + 1
        else:
            max_k = mid_k - 1

    return best_kmeans


def flatten_values_lists_of_list_dicts_to_dict(item):
    """
    Flattens a list of dictionaries containing lists of values into a single dictionary.

    Args:
        item (list): List of dictionaries, where each dictionary's values are lists. If any element of the list is itself a list, the function will consider only the first dictionary in that sublist.

    Returns:
        dict: A dictionary where each key corresponds to the keys in the input dictionaries, and each value is a flattened list of all values associated with that key across all input dictionaries.
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

        if not local_references is None:
            dist.gather_object(local_references, global_references_list, dst=0)

    except Exception as e:
        print(f"Error during result gathering: {e}")

    if local_rank != 0:
        # Exit the process
        # print(f"Rank {dist.get_rank()} exiting.")
        dist.destroy_process_group()  # Clean up the distributed processing group
        sys.exit()  # Exit the process

    # Flatten the gathered list
    candidates_list = []
    for i in global_candidates_list:
        candidates_list.extend(i)

    if not global_references_list[0] is None:
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
        if (
            "[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>"
        ) in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def make_prompt(text1, text2, max_len=300):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in comparison to a reference radiology report.

    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.

    Returns:
        str: Formatted prompt string.
    """
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
    return prompt
