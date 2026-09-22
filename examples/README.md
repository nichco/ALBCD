# Examples

Every example is included twice: in `jax/` with gradients from JAX, and in
`pytorch/` with gradients from PyTorch. The block subproblems are solved with
[modopt](https://github.com/LSDOlab/modopt). Install the dependencies from the
repository root with

```bash
pip install -e ".[examples]"
```

and run any example directly, e.g. `python examples/pytorch/rosenbrock_consensus.py`.

Two larger wing design examples, in PyTorch only, are in
[pytorch/aerostruct/](pytorch/aerostruct/README.md) (aerostructural design) and
[pytorch/uCRM/](pytorch/uCRM/README.md) (multipoint design).