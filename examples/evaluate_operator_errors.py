import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlspike import NLSpikeRMSNorm, NLSpikeSiLU, NLSpikeSoftmax, quantize_floor


def evaluate_silu(device: str, n_points: int, q_bits: int, silu_bound: float):
    x = torch.linspace(-silu_bound, silu_bound, n_points, device=device)
    ref = quantize_floor(F.silu(x), q_bits=q_bits)
    ours = NLSpikeSiLU(q_bits=q_bits).to(device)(x)
    err = (ours - ref).abs()
    return {
        "operator": "silu",
        "dim": 1,
        "max_abs_error": err.max().item(),
        "mean_abs_error": err.mean().item(),
    }


def evaluate_softmax(device: str, dims, n_vectors: int, q_bits: int):
    rows = []
    op = NLSpikeSoftmax(dim=-1, q_bits=q_bits).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    for dim in dims:
        logits = torch.randn(n_vectors, dim, device=device, generator=generator) * 7.0
        ref = quantize_floor(torch.softmax(logits, dim=-1), q_bits=q_bits)
        ours = op(logits)
        err = (ours - ref).abs()
        rows.append(
            {
                "operator": "softmax",
                "dim": dim,
                "max_abs_error": err.max().item(),
                "mean_abs_error": err.mean().item(),
            }
        )
    return rows


def evaluate_rmsnorm(device: str, dims, n_vectors: int, q_bits: int):
    rows = []
    generator = torch.Generator(device=device)
    generator.manual_seed(1)
    for dim in dims:
        x = torch.randn(n_vectors, dim, device=device, generator=generator)
        ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        ref = quantize_floor(ref, q_bits=q_bits)
        ours = NLSpikeRMSNorm(hidden_size=None, q_bits=q_bits).to(device)(x)
        err = (ours - ref).abs()
        rows.append(
            {
                "operator": "rmsnorm",
                "dim": dim,
                "max_abs_error": err.max().item(),
                "mean_abs_error": err.mean().item(),
            }
        )
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["operator", "dim", "max_abs_error", "mean_abs_error"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dims", default="8,16,32,64,128,256")
    parser.add_argument("--n_vectors", type=int, default=4096)
    parser.add_argument("--n_points", type=int, default=20001)
    parser.add_argument("--silu_bound", type=float, default=5.0)
    parser.add_argument("--q_bits", type=int, default=8)
    parser.add_argument("--output", default="results/operator_errors.csv")
    args = parser.parse_args()

    dims = [int(item) for item in args.dims.split(",") if item]
    rows = [evaluate_silu(args.device, args.n_points, args.q_bits, args.silu_bound)]
    rows.extend(evaluate_softmax(args.device, dims, args.n_vectors, args.q_bits))
    rows.extend(evaluate_rmsnorm(args.device, dims, args.n_vectors, args.q_bits))

    write_csv(rows, Path(args.output))
    for row in rows:
        print(
            f"{row['operator']:8s} dim={row['dim']:4d} "
            f"max={row['max_abs_error']:.6e} mean={row['mean_abs_error']:.6e}"
        )
    print(f"Saved results to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
