# ml_example_project

A starter workspace for Jupyter notebooks and machine learning experiments.

## Environment Setup

This project uses a conda environment defined in `environment.yml`.

1. Create the conda environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate it:

   ```bash
   conda activate ml_example_project
   ```

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

When Jupyter Notebook starts from the activated conda environment, use the
Python kernel from `ml_example_project`.

If you want this environment to appear as a named kernel in other Jupyter
installations, register it with:

```bash
python -m ipykernel install --user --name ml_example_project --display-name "Python (ml_example_project)"
```

Then choose `Python (ml_example_project)` when opening a notebook.

To update the environment after changing `environment.yml`, run:

```bash
conda env update -f environment.yml --prune
```

To remove the environment, run:

```bash
conda env remove -n ml_example_project
```

## Included Packages

- Jupyter Notebook, JupyterLab, and IPython kernel support
- NumPy, pandas, and SciPy for data work
- scikit-learn for machine learning
- Matplotlib, seaborn, and Plotly for visualization
- python-dotenv for local environment variables
- ucimlrepo for loading datasets from the UCI Machine Learning Repository
