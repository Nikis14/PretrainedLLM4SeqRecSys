import pandas as pd
import json
import umap
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def preprocess_null_field(field: Any) -> str:
    """Converts a field to a string, replacing null values with 'null'.

    Args:
        field (Any): The field to preprocess.

    Returns:
        str: The processed field as a string.
    """
    return str(field) if not pd.isnull(field) else "null"


def read_json(file_path: str) -> Dict:
    """Reads a JSON file and returns its content.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict: The data loaded from the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(file_path: str) -> str:
    """Reads a text file and returns its content.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_json(data: Dict, file_path: str) -> None:
    """Saves a dictionary as a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def visualize_embeddings(
    json_file_path: str, output_file_path="umap.png", annotate_points=False
):
    """
    Loads user embeddings from a JSON file, reduces dimensionality using UMAP, and plots the results.

    Args:
        data (Dict): The data to save.
        file_path (str): The path to the output JSON file.
        json_file_path (str): The path to the JSON file containing user embeddings.
    """
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
        return

    embeddings = list(data.values())

    if not all(isinstance(emb, list) for emb in embeddings):
        print("Error: Embeddings must be lists of numbers.")
        return
    if not all(all(isinstance(val, (int, float)) for val in emb) for emb in embeddings):
        print("Error: Embeddings must contain only numbers.")
        return

    embeddings_np = np.array(embeddings)

    reducer = umap.UMAP(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings_np)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=20)

    # Annotate points with user IDs
    if annotate_points:
        user_ids = list(data.keys())
        for i, user_id in enumerate(user_ids):
            plt.annotate(user_id, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    plt.title("UMAP Projection of User Embeddings")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.savefig(output_file_path)
    plt.close()


def cosine(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculates cosine similarity between two embeddings.

    Args:
        embedding1 (np.ndarray): The first embedding.
        embedding2 (np.ndarray): The second embedding.

    Returns:
        float: The cosine similarity between the two embeddings.
    """
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


def calculate_distance_pairs(
    json_path: str,
    max_users: int = 50000,
    metric: str = "cosine",
) -> Dict[str, Any]:
    """Calculates cosine similarity distance between pairs of embeddings.

    Args:
        json_path (str): The path to the JSON file containing user embeddings.
        max_users (int, optional): The maximum number of users to consider. Defaults to 50000.
        metric (str, optional): The distance metric to use. Defaults to "cosine".

    Returns:
        Dict[str, Any]: A dictionary with the minimum distance, maximum distance, and average distance.
    """
    with open(json_path, "r") as f:
        user_data = json.load(f)

    user_ids = np.array(list(user_data.keys()))
    embeddings = np.array(list(user_data.values()))

    if len(user_ids) > max_users:
        sampled_indices = np.random.choice(len(user_ids), size=max_users, replace=False)
        user_ids = user_ids[sampled_indices]
        embeddings = embeddings[sampled_indices]

    if metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = embeddings / norms

        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    num_users = len(user_ids)
    i_indices, j_indices = np.triu_indices(num_users, k=1)
    user_id1 = user_ids[i_indices]
    user_id2 = user_ids[j_indices]
    distances = similarity_matrix[i_indices, j_indices]

    min_distance_idx = np.argmin(distances)
    max_distance_idx = np.argmax(distances)
    min_distance = distances[min_distance_idx]
    max_distance = distances[max_distance_idx]
    avg_distance = np.mean(distances)

    min_distance_info = {
        "user_id1": user_id1[min_distance_idx],
        "user_id2": user_id2[min_distance_idx],
        "distance": min_distance,
    }

    max_distance_info = {
        "user_id1": user_id1[max_distance_idx],
        "user_id2": user_id2[max_distance_idx],
        "distance": max_distance,
    }

    summary = {
        "min_distance": min_distance_info,
        "max_distance": max_distance_info,
        "average_distance": avg_distance,
    }

    return summary
