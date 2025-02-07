import numpy as np
import choix
from scipy import stats
import warnings
import pandas as pd


def convert_matrix_to_comparisons(comparison_matrix):
    """
    Convert a comparison matrix to a list of (winner, loser) pairs.
    The matrix values are first rescaled from [0,1] to [0.33,0.66].

    Parameters:
    comparison_matrix (np.ndarray): An N x N matrix where entry (i,j) represents
                                  win probability of i over j

    Returns:
    list: List of (winner, loser) tuples, with weighted entries
    """

    def rescale_to_33_66(value):
        """Rescale a value from [0,1] to [0.33,0.66]"""
        return 0.33 + (0.66 - 0.33) * value

    comparisons = []
    n_players = len(comparison_matrix)

    for i in range(n_players):
        for j in range(n_players):
            if i != j:  # Skip self-comparisons
                # Scale the probability to [0.33, 0.66]
                rescaled_prob = rescale_to_33_66(comparison_matrix[i, j])

                # Convert to integer number of comparisons
                # Multiply by 100 to preserve decimal precision
                n_comparisons = int(round(rescaled_prob * 100))

                # Add the pairs to the comparison list
                comparisons.extend([(i, j)] * n_comparisons)

    return comparisons


def calculate_standard_errors(parameters, comparison_matrix):
    """
    Calculate standard errors for Bradley-Terry model parameters using Fisher Information.

    Parameters:
    parameters (np.ndarray): Model parameters (log-skills)
    comparison_matrix (np.ndarray): Original comparison matrix

    Returns:
    np.ndarray: Standard errors for each parameter
    """
    n_players = len(parameters)
    exp_params = np.exp(parameters)
    total_games = comparison_matrix + comparison_matrix.T

    # Initialize Fisher Information Matrix
    fim = np.zeros((n_players, n_players))

    # Calculate Fisher Information Matrix
    for i in range(n_players):
        for j in range(n_players):
            if i != j and total_games[i, j] > 0:
                p_ij = exp_params[i] / (exp_params[i] + exp_params[j])
                fim[i, i] += total_games[i, j] * p_ij * (1 - p_ij)
                fim[i, j] -= total_games[i, j] * p_ij * (1 - p_ij)
                fim[j, j] += total_games[i, j] * p_ij * (1 - p_ij)

    # Remove last row and column (for identifiability)
    fim = fim[:-1, :-1]

    try:
        # Calculate inverse of Fisher Information Matrix
        fim_inv = np.linalg.inv(fim)
        # Add zero for last parameter (reference category)
        std_errors = np.sqrt(np.diag(fim_inv))
        std_errors = np.append(std_errors, 0)
        return std_errors
    except np.linalg.LinAlgError:
        warnings.warn(
            "Fisher Information Matrix is singular. Standard errors may not be reliable."
        )
        return np.full(n_players, np.nan)


def rescale_matrix_values(matrix):
    """
    Rescale entire matrix from [0,1] to [33,66] while maintaining 50 as midpoint.

    Parameters:
    matrix (np.ndarray): Original comparison matrix with values in [0,1]

    Returns:
    np.ndarray: Rescaled matrix with values in [33, 66]
    """
    new_min = 33
    new_max = 66

    # Linear rescaling: new = a * old + b
    a = new_max - new_min  # Range in new scale
    b = new_min  # Offset

    # Apply rescaling to entire matrix
    rescaled = a * matrix + b

    return rescaled


def fit_bradley_terry(comparison_matrix, method="lsr", alpha=0.05):
    """
    Fit Bradley-Terry model using choix library and calculate statistical measures.
    Now includes connectivity check, perturbation if needed, and rating rescaling.
    """
    # comparison_matrix = rescale_matrix_values(comparison_matrix)
    # Check directed connectivity

    n_players = len(comparison_matrix)
    comparisons = convert_matrix_to_comparisons(comparison_matrix)

    # Fit the model using specified method
    if method == "lsr":
        parameters = choix.ilsr_pairwise(n_players, comparisons)
    elif method == "mm":
        parameters = choix.mm_pairwise(n_players, comparisons)
    elif method == "opt":
        parameters = choix.opt_pairwise(n_players, comparisons)
    else:
        raise ValueError("Method must be one of: 'lsr', 'mm', 'opt'")

    # Calculate standard errors
    std_errors = calculate_standard_errors(parameters, comparison_matrix)

    # Calculate confidence intervals
    z_value = stats.norm.ppf(1 - alpha / 2)
    conf_intervals = np.array(
        [parameters - z_value * std_errors, parameters + z_value * std_errors]
    )

    # Calculate p-values
    z_scores = parameters / std_errors
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # Convert parameters to probability scale
    exp_params = np.exp(parameters)
    original_ratings = exp_params / np.sum(exp_params)

    return {
        "parameters": parameters,
        "ratings": original_ratings,  # ratings
        "std_errors": std_errors,
        "conf_intervals": conf_intervals,
        "p_values": p_values,
    }


def get_win_probability(params, player_a, player_b):
    """
    Calculate probability of player_a beating player_b given model parameters.

    Parameters:
    params (np.ndarray): Model parameters (log-skills)
    player_a (int): Index of first player
    player_b (int): Index of second player

    Returns:
    float: Probability of player_a winning against player_b
    """
    return choix.probabilities([player_a, player_b], params)[0]


def check_directed_connectivity(comparison_matrix):
    """
    Check if the comparison matrix represents a strongly connected directed graph.
    Returns True if strongly connected, False otherwise.
    """
    n = len(comparison_matrix)

    # Create adjacency list representation
    graph = [[] for _ in range(n)]

    # Build adjacency lists (only add edge if there's at least one comparison)
    for i in range(n):
        for j in range(n):
            if comparison_matrix[i][j] > 0:
                graph[i].append(j)

    # Helper function for DFS
    def dfs(v, visited):
        visited[v] = True
        for u in graph[v]:
            if not visited[u]:
                dfs(u, visited)

    # Check if all vertices are reachable from each vertex
    for start in range(n):
        visited = [False] * n
        dfs(start, visited)
        if not all(visited):
            return False

    return True


def create_perturbed_matrix(comparison_matrix, epsilon=1):
    """
    Create perturbed adjacency matrix following equation (2) in the paper:
    ãᵢⱼ = aᵢⱼ + εI(nᵢⱼ > 0 or nⱼᵢ > 0)
    Matrix is first multiplied by 10 to handle decimal values.
    """
    # Scale the matrix first
    scaled_matrix = comparison_matrix * 10

    # Create indicator matrix for any comparison between i and j
    indicator = ((scaled_matrix + scaled_matrix.T) > 0).astype(float)

    # Apply perturbation: ãᵢⱼ = aᵢⱼ + εI(nᵢⱼ > 0 or nⱼᵢ > 0)
    perturbed_matrix = scaled_matrix + epsilon * indicator

    return perturbed_matrix


def print_and_save_results(results, method, comparison_matrix):
    """
    Print detailed results and save rankings to CSV using original image IDs.
    Includes the following columns:
    - image_id
    - complexity
    - matrix_index
    - parameter
    - std_error
    - ci_lower
    - ci_upper
    - p_value
    """
    # Load the index to ID mapping
    id_mapping = np.load("npy/id_to_index.npy", allow_pickle=True).item()
    index_to_id = {v: k for k, v in id_mapping.items()}

    # Create DataFrame with required columns
    data = []
    for i in range(len(comparison_matrix)):
        original_id = index_to_id[i]
        data.append(
            {
                "image_id": int(original_id),
                "complexity": results["ratings"][i],
                "matrix_index": i,
                "parameter": results["parameters"][i],
                "std_error": results["std_errors"][i],
                "ci_lower": results["conf_intervals"][0][i],
                "ci_upper": results["conf_intervals"][1][i],
                "p_value": results["p_values"][i],
            }
        )

    df = pd.DataFrame(data)

    # Sort by complexity in descending order
    df = df.sort_values("complexity", ascending=False)

    # Save to CSV
    filename = f"bradley_terry_complexity_{method}.csv"
    df.to_csv(filename, index=False)
    print(f"\nComplexity scores saved to {filename}")

    return df


def example_usage():
    # Load comparison matrix
    comparison_matrix = np.load("npy/probability_matrix.npy")

    # Fit model using different methods
    methods = ["lsr", "mm", "opt"]

    all_rankings = {}

    for method in methods:
        results = fit_bradley_terry(comparison_matrix, method=method)
        rankings_df = print_and_save_results(results, method, comparison_matrix)
        all_rankings[method] = rankings_df

    return all_rankings


if __name__ == "__main__":
    example_usage()
