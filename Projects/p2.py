import math
import random
from collections import Counter


FEATURE_NAMES = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
]

CLASS_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica",
}


def load_dataset():
    return [
        ([5.1, 3.5, 1.4, 0.2], 0), ([4.9, 3.0, 1.4, 0.2], 0),
        ([4.7, 3.2, 1.3, 0.2], 0), ([4.6, 3.1, 1.5, 0.2], 0),
        ([5.0, 3.6, 1.4, 0.2], 0), ([5.4, 3.9, 1.7, 0.4], 0),
        ([4.6, 3.4, 1.4, 0.3], 0), ([5.0, 3.4, 1.5, 0.2], 0),
        ([7.0, 3.2, 4.7, 1.4], 1), ([6.4, 3.2, 4.5, 1.5], 1),
        ([6.9, 3.1, 4.9, 1.5], 1), ([5.5, 2.3, 4.0, 1.3], 1),
        ([6.5, 2.8, 4.6, 1.5], 1), ([5.7, 2.8, 4.5, 1.3], 1),
        ([6.3, 3.3, 4.7, 1.6], 1), ([4.9, 2.4, 3.3, 1.0], 1),
        ([6.3, 3.3, 6.0, 2.5], 2), ([5.8, 2.7, 5.1, 1.9], 2),
        ([7.1, 3.0, 5.9, 2.1], 2), ([6.3, 2.9, 5.6, 1.8], 2),
        ([6.5, 3.0, 5.8, 2.2], 2), ([7.6, 3.0, 6.6, 2.1], 2),
        ([4.9, 2.5, 4.5, 1.7], 2), ([7.3, 2.9, 6.3, 1.8], 2),
    ]


def print_dataset_summary(dataset):
    class_counts = Counter(label for _, label in dataset)

    print("Dataset loaded: Iris flower classification")
    print(f"Total samples: {len(dataset)}")
    print(f"Features: {', '.join(FEATURE_NAMES)}")

    print("\nClass distribution:")
    for label, name in CLASS_NAMES.items():
        print(f"{name:<12}: {class_counts[label]} samples")

    print("\nFirst 5 records:")
    for index, (features, label) in enumerate(dataset[:5], start=1):
        feature_text = ", ".join(
            f"{name}={value:.1f}" for name, value in zip(FEATURE_NAMES, features)
        )
        print(f"{index}. {feature_text} -> {CLASS_NAMES[label]}")


def train_test_split(dataset, test_count=16):
    shuffled_data = dataset[:]
    random.seed(5)
    random.shuffle(shuffled_data)

    test_data = shuffled_data[:test_count]
    train_data = shuffled_data[test_count:]

    return train_data, test_data


def euclidean_distance(first, second):
    squared_differences = [
        (first[index] - second[index]) ** 2 for index in range(len(first))
    ]
    return math.sqrt(sum(squared_differences))


def predict_knn(train_data, sample, k=3):
    distances = []

    for features, label in train_data:
        distance = euclidean_distance(features, sample)
        distances.append((distance, label))

    nearest_neighbors = sorted(distances)[:k]
    votes = Counter(label for _, label in nearest_neighbors)
    return votes.most_common(1)[0][0]


def evaluate_model(train_data, test_data, k=3):
    correct = 0

    print("\nTesting predictions:")
    print("No.  Actual       Predicted    Result")
    print("----------------------------------------")

    for index, (features, actual_label) in enumerate(test_data, start=1):
        predicted_label = predict_knn(train_data, features, k)
        result = "Correct" if predicted_label == actual_label else "Wrong"

        if predicted_label == actual_label:
            correct += 1

        print(
            f"{index:<4} {CLASS_NAMES[actual_label]:<12} "
            f"{CLASS_NAMES[predicted_label]:<12} {result}"
        )

    return correct / len(test_data)


def main():
    dataset = load_dataset()
    print_dataset_summary(dataset)

    train_data, test_data = train_test_split(dataset)

    print("\nData split complete:")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")

    print("\nModel: K-Nearest Neighbors classifier, k = 3")
    accuracy = evaluate_model(train_data, test_data, k=3)
    print(f"\nModel accuracy: {accuracy * 100:.0f}%")

    new_sample = [5.1, 3.5, 1.4, 0.2]
    prediction = predict_knn(train_data, new_sample, k=3)

    print("\nExample new sample prediction:")
    print(f"{new_sample} -> {CLASS_NAMES[prediction]}")


if __name__ == "__main__":
    main()
