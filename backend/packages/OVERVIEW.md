# backend/packages/ Overview

## What Is This?

The `packages/` folder contains **vendored third-party code** that is bundled directly with the backend rather than installed from PyPI. Currently it holds one package:

```
packages/
└── cnmf/   # Local copy of the cNMF library (Kotliar et al., 2019)
```

---

## `packages/cnmf/`

### What is cNMF?

**cNMF** (consensus Non-negative Matrix Factorization) is an open-source Python library developed by Dylan Kotliar et al. for discovering gene expression programs from single-cell and bulk RNA-seq data.

- **Paper:** Kotliar et al., *eLife* 2019 — ["Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-seq"](https://elifesciences.org/articles/43803)
- **Original repo:** [github.com/dylkot/cNMF](https://github.com/dylkot/cNMF)
- This is a **git submodule** — the `.git/` folder inside it points to the upstream repository

### Why is it vendored here?

Vendoring allows pinning to a specific version and applying any custom modifications without depending on the upstream package being available on PyPI at the exact right version.

### Structure

```
cnmf/
├── src/
│   └── cnmf/
│       ├── cnmf.py        # Core cNMF class and factorization logic
│       └── ...
├── tests/
├── Tutorials/
└── Extras/
```

### How It's Used in This Codebase

The `backend/nmf/cNMF.py` module imports from this vendored package:

```python
from cnmf import cNMF
```

`cNMF.py` uses the `cNMF` class to:
1. Initialize a factorization run with a chosen `k`
2. Distribute factorization across 16 parallel workers
3. Run the consensus step to finalize stable gene modules

See `backend/nmf/OVERVIEW.md` for details on how the consensus NMF pipeline works.

### Updating the Package

Since this is a git submodule, to update it:
```bash
cd backend/packages/cnmf
git pull origin main
```

Do not modify files inside `packages/cnmf/` directly unless you intend to diverge from upstream.
