import random
import csv
from collections import defaultdict
import numpy as np
import networkx as nx


def load_images(filename):
    """Load image IDs from the input file."""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_base_comparisons(images):
    """Create base comparisons that ensure connectivity (1-2, 2-3, etc.)"""
    base_comparisons = []
    for i in range(len(images) - 1):
        base_comparisons.append((images[i], images[i + 1]))
    # Connect last image to first to ensure strong connectivity
    base_comparisons.append((images[-1], images[0]))
    return base_comparisons


def distribute_base_comparisons(base_comparisons, num_sessions):
    """Distribute base comparisons evenly across sessions"""
    distributed_comparisons = defaultdict(list)
    for idx, (img1, img2) in enumerate(base_comparisons):
        session_id = (idx % num_sessions) + 1
        distributed_comparisons[session_id].append([session_id, img1, img2])
    return distributed_comparisons


def generate_comparisons(
    images,
    num_sessions=400,
    unique_comparisons_per_session=250,
    repetitions_per_session=5,
    attention_checks_per_session=4,
):
    """Generate pairwise comparisons ensuring strong connectivity and even distribution."""
    # Create and distribute base comparisons
    base_comparisons = create_base_comparisons(images)
    session_comparisons = distribute_base_comparisons(base_comparisons, num_sessions)

    # Track usage count for weighted random selection
    usage_count = defaultdict(int)
    for session_list in session_comparisons.values():
        for _, img1, img2 in session_list:
            usage_count[img1] += 1
            usage_count[img2] += 1

    # Fill each session with random comparisons until reaching target
    for session in range(1, num_sessions + 1):
        current_unique = len(session_comparisons[session])
        needed_unique = (
            unique_comparisons_per_session
            - attention_checks_per_session
            - current_unique
        )

        # Generate additional random comparisons for this session
        while needed_unique > 0:
            # Weighted random selection based on usage count
            weights = [1 / (usage_count[img] + 1) for img in images]
            weights = np.array(weights) / sum(weights)
            img1, img2 = np.random.choice(images, size=2, replace=False, p=weights)

            # Add new comparison
            session_comparisons[session].append([session, img1, img2])
            usage_count[img1] += 1
            usage_count[img2] += 1
            needed_unique -= 1

        # Add repetitions
        original_comparisons = session_comparisons[session].copy()
        repeats = random.sample(original_comparisons, repetitions_per_session)
        for repeat in repeats:
            insert_position = random.randint(0, len(session_comparisons[session]))
            session_comparisons[session].insert(insert_position, repeat)

        # Add attention checks
        for _ in range(attention_checks_per_session):
            if random.random() < 0.5:
                attention_check = [session, "0", "-1"]
            else:
                attention_check = [session, "-1", "0"]
            insert_position = random.randint(0, len(session_comparisons[session]))
            session_comparisons[session].insert(insert_position, attention_check)

    # Flatten all comparisons
    all_comparisons = []
    for session_list in session_comparisons.values():
        all_comparisons.extend(session_list)

    # Verify connectivity
    G = nx.DiGraph()
    for session_id, img1, img2 in all_comparisons:
        if img1 not in ["-1", "0"] and img2 not in ["-1", "0"]:
            G.add_edge(img1, img2)
            G.add_edge(img2, img1)

    if nx.is_strongly_connected(G):
        print("Success: Graph is strongly connected!")
    else:
        print("Warning: Graph is not strongly connected!")

    return all_comparisons


def save_comparisons(comparisons, output_file):
    """Save comparisons to a CSV file."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "image1", "image2"])
        writer.writerows(comparisons)


def print_statistics(
    comparisons, images, repetitions_per_session, attention_checks_per_session
):
    """Print statistics about image usage, repetitions, and attention checks."""
    usage_count = defaultdict(int)
    sessions = defaultdict(list)
    attention_check_counts = defaultdict(int)
    unique_comparisons_per_session = defaultdict(set)

    for session_id, img1, img2 in comparisons:
        if img1 not in ["0", "-1"] and img2 not in ["0", "-1"]:
            usage_count[img1] += 1
            usage_count[img2] += 1
            unique_comparisons_per_session[session_id].add((img1, img2))
        else:
            attention_check_counts[session_id] += 1
        sessions[session_id].append((img1, img2))

    print("\nImage Usage Statistics:")
    print(f"Total images: {len(images)}")
    print(
        f"Average appearances per image: {sum(usage_count.values()) / len(images):.2f}"
    )
    print(f"Min appearances: {min(usage_count.values())}")
    print(f"Max appearances: {max(usage_count.values())}")

    print("\nSession Statistics:")
    print(f"Total sessions: {len(sessions)}")
    print(f"Target repetitions per session: {repetitions_per_session}")
    print(f"Target attention checks per session: {attention_checks_per_session}")
    print("\nUnique comparisons per session:")
    for session_id in sorted(unique_comparisons_per_session.keys()):
        print(
            f"Session {session_id}: {len(unique_comparisons_per_session[session_id])}"
        )

    print("\nAttention Check Statistics:")
    attention_counts = list(attention_check_counts.values())
    print(f"Average attention checks per session: {np.mean(attention_counts):.2f}")
    print(f"Min attention checks in a session: {min(attention_counts)}")
    print(f"Max attention checks in a session: {max(attention_counts)}")


def main():
    # Configuration
    input_file = "sample.txt"
    output_file = "comparisons_30.csv"
    num_sessions = 30  # 400
    unique_comparisons_per_session = 200  # 250
    repetitions_per_session = 3  # 5
    attention_checks_per_session = 2  # 4

    # Load images
    images = load_images(input_file)

    # Generate comparisons
    print("Generating comparisons...")
    comparisons = generate_comparisons(
        images,
        num_sessions,
        unique_comparisons_per_session,
        repetitions_per_session,
        attention_checks_per_session,
    )

    # Save results
    save_comparisons(comparisons, output_file)
    total_comparisons = len(comparisons)
    regular_comparisons = total_comparisons - (
        num_sessions * attention_checks_per_session
    )
    print(f"\nGenerated {total_comparisons} total comparisons")
    print(f"Regular comparisons: {regular_comparisons}")
    print(f"Attention checks: {num_sessions * attention_checks_per_session}")
    print(f"Total sessions: {num_sessions}")
    print(f"Saved to: {output_file}")

    # Print statistics
    print_statistics(
        comparisons, images, repetitions_per_session, attention_checks_per_session
    )


if __name__ == "__main__":
    main()
