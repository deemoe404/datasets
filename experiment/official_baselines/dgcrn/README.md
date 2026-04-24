# DGCRN Official-Core Wrapper

- Source: `https://github.com/tsinghua-fib-lab/Traffic-Benchmark.git`
- Pinned commit: `b9f8e40b4df9b58f5ad88432dc070cbbbcdc0228`
- Source path: `source/`
- License: MIT, root `source/LICENSE`
- Adapter target: `world_model_dgcrn_official_core_v1_farm_sync`

The official Traffic-Benchmark repository is retained as the source-of-truth
reference for DGCRN. The hardened family keeps the repository's dynamic graph
recurrent architecture as the conceptual contract while adapting inputs through
the existing `world_model_v1` farm-synchronous task bundle.

Run or refresh the wrapper environment with:

```shell
cd experiment/official_baselines/dgcrn
./create_env.sh
```
