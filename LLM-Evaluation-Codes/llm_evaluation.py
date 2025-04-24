import numpy as np

def llm_consensus_evaluation(outputs: np.ndarray):
    """
    Evaluate LLM consensus based on pairwise L2 distance.

    Parameters:
    - outputs: np.ndarray of shape (m, n), where
        - m is the number of LLMs
        - n is the number of classification samples
        - Each row is an LLM's output: values in {1,2,3,4,5}

    Returns:
    - D: np.ndarray of shape (m,), raw deviation scores
    - D_tilde: np.ndarray of shape (m,), normalized deviation scores
    - optimal_llm_index: int, index of the most aligned (least deviating) LLM
    """
    m, n = outputs.shape
    pairwise_distances = np.zeros((m, m))

    # Step 1: Compute pairwise L2 distances
    for j in range(m):
        for k in range(j + 1, m):
            dist = np.linalg.norm(outputs[j] - outputs[k])
            pairwise_distances[j, k] = dist
            pairwise_distances[k, j] = dist  # symmetric

    # Step 2: Compute mean deviation scores
    D = np.zeros(m)
    for j in range(m):
        D[j] = np.sum(pairwise_distances[j]) / (m - 1)

    # Step 3: Normalize deviation scores
    D_tilde = D / np.max(D)

    # The optimal LLM is the one with the lowest normalized deviation score
    optimal_llm_index = int(np.argmin(D_tilde))

    return D, D_tilde, optimal_llm_index
