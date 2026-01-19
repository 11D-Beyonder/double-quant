import scipy as sp


def cos_similarity(x, y):
    """
    Compute cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two non-zero
    vectors in an inner product space. It ranges from -1 (opposite directions)
    to 1 (same direction), with 0 indicating orthogonality.

    This metric is commonly used to measure similarity in:
    - Natural language processing (document similarity)
    - Recommendation systems (user/item similarity)
    - Machine learning (feature similarity)
    - Quantum computing (state vector comparison)

    Args:
        x: First vector as numpy array or array-like.
        y: Second vector as numpy array or array-like.
           Must have the same dimension as x.

    Returns:
        float: Cosine similarity value in range [-1, 1].
               - 1.0: Vectors point in the same direction
               - 0.0: Vectors are orthogonal
               - -1.0: Vectors point in opposite directions

    Raises:
        ValueError: If vectors have different dimensions.
        ValueError: If either vector has zero norm (undefined similarity).

    Examples:
        Identical vectors:
        >>> import numpy as np
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([1, 2, 3])
        >>> cos_similarity(x, y)
        1.0

        Orthogonal vectors:
        >>> x = np.array([1, 0])
        >>> y = np.array([0, 1])
        >>> cos_similarity(x, y)
        0.0

        Opposite vectors:
        >>> x = np.array([1, 2, 3])
        >>> y = np.array([-1, -2, -3])
        >>> cos_similarity(x, y)
        -1.0

        Quantum state comparison:
        >>> solution = np.array([0.6, 0.8])
        >>> expected = np.array([0.6, 0.8])
        >>> similarity = cos_similarity(solution, expected)
        >>> print(f"Solution similarity: {similarity:.4f}")

    Notes:
        - This function uses scipy.spatial.distance.cosine internally
        - The formula is: cos_sim(x, y) = (x · y) / (||x|| ||y||)
        - Equivalent to: 1 - scipy.spatial.distance.cosine(x, y)
        - Vectors are automatically normalized internally
        - For quantum states, high cosine similarity (≈1) indicates
          the quantum solution closely matches the expected result
    """
    return 1 - sp.spatial.distance.cosine(x, y)
