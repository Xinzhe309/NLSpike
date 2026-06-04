# NLSpike Float Reference

This repository contains a portable float reference implementation of NLSpike
nonlinear operator replacements for Transformer models.

NLSpike targets the nonlinear operators that are hard to express directly in
ANN-to-SNN conversion pipelines:

- Softmax
- SiLU
- RMSNorm

The implementation here is intentionally independent of a specific SNN runtime.
It is meant to reproduce operator-level approximation behavior and to serve as a
clean reference for plugging NLSpike-style replacements into ANN, SpikeLLM, or
other ANN-to-SNN workflows.

## Scope

This release provides the float reference implementation only. It does not claim
to be a hardware backend or a universal LIF neuron implementation.

Concrete LIF classes differ across SNN frameworks and LLM conversion pipelines.
For that reason, backend-specific spike implementations should be built as thin
adapters around the same numerator, denominator, PWL-Exp, and PolarNorm
decomposition used by this reference.

## Repository Layout

```text
.
|-- nlspike/
|   |-- __init__.py
|   `-- ops.py
|-- examples/
|   `-- evaluate_operator_errors.py
|-- LICENSE
|-- README.md
|-- requirements.txt
`-- pyproject.toml
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
import torch
from nlspike import NLSpikeSiLU, NLSpikeSoftmax, NLSpikeRMSNorm

x = torch.randn(2, 128)
logits = torch.randn(2, 16)

silu = NLSpikeSiLU()
softmax = NLSpikeSoftmax(dim=-1)
rmsnorm = NLSpikeRMSNorm(hidden_size=128)

y_silu = silu(x)
y_softmax = softmax(logits)
y_rmsnorm = rmsnorm(x)
```

## Operator Evaluation

Run the included random-tensor evaluation script:

```bash
python examples/evaluate_operator_errors.py
```

The script evaluates SiLU, Softmax, and RMSNorm approximations against PyTorch
float references quantized to the same output grid. By default, SiLU is tested
on `[-5, 5]`, matching the clipped interval used by the reference setting. It
writes results to:

```text
results/operator_errors.csv
```

No datasets, model checkpoints, or pretrained weights are required.

## Method Summary

NLSpike approximates Transformer nonlinearities through reusable operator
primitives:

- PWL-Exp: piecewise-linear exponential approximation on a clipped interval
- Division-style normalization: float reference for the quotient stage
- PolarNorm: CORDIC-style L2 norm approximation for RMSNorm

The float reference keeps the same operator-level decomposition while avoiding
any dependency on one framework's LIF state, reset, threshold, or tensor-layout
conventions.

## What Is Not Included

This release intentionally excludes:

- datasets and cached dataloaders
- pretrained weights and quantized checkpoints
- training logs and generated experiment artifacts
- backend-specific SpikeLLM or SpikeZIP LIF classes
- hardware-specific kernels
- private paths or machine-specific scripts

## Citation

If you use this reference implementation, please cite the NLSpike paper once the
public citation information is available.

## License

This project is released under the MIT License. See `LICENSE` for details.
