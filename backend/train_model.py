from __future__ import annotations

from app.model_service import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, train_and_save_model


def main() -> None:
    artifact = train_and_save_model(DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH)
    print(f"Model saved to: {DEFAULT_MODEL_PATH}")
    threshold = artifact["metrics"].get("decision_threshold")
    if threshold is not None:
        print(f"Best threshold (search 0.25-0.45, recall >= 0.70): {float(threshold):.4f}")
    print("Model metrics:")
    for metric, value in artifact["metrics"].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()

