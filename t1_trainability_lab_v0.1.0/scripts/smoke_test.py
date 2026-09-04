"""Minimal CPU-only PyTorch environment smoke test."""

import torch


def main() -> None:
    if torch.version.cuda is not None:
        raise RuntimeError(f"Expected CPU-only PyTorch, got CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        raise RuntimeError("Expected CUDA to be unavailable")

    tensor = torch.zeros(1)
    if tensor.device.type != "cpu":
        raise RuntimeError(f"Expected CPU tensor, got {tensor.device}")

    print(f"torch={torch.__version__}")
    print(f"device={tensor.device}")
    print(f"tensor={tensor}")


if __name__ == "__main__":
    main()
