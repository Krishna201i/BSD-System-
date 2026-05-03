from __future__ import annotations

from app.model_service import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, train_and_save_model


def main() -> None:
    artifact = train_and_save_model(DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH)
    print(f"Model saved to: {DEFAULT_MODEL_PATH}")
    print("Model metrics:")
    for metric, value in artifact["metrics"].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()

