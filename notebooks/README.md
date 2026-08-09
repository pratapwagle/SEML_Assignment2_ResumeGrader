# Notebooks

| File | Role |
|------|------|
| `179.ipynb` | Canonical assignment notebook (`<group>.ipynb` naming) |
| `179_executed.ipynb` | Same notebook with saved outputs (execution evidence) |

Both notebooks live under `notebooks/`. A setup cell sets the working directory
to the repository root so imports (`ml`, `app`, `services`) and paths such as
`models/` and `report/` resolve correctly.

```bash
# from repository root
jupyter notebook notebooks/179.ipynb

# or from notebooks/
cd notebooks
jupyter notebook 179.ipynb
```
