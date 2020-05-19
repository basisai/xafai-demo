# xai-fairness
Toolkit for model explainability and fairness

## Instructions
There are three files that require user's inputs.
- `xai_fairness.yaml`
- `constants.py`
- `report_utils.py`

### `xai_fairness.yaml`
- This file will provide the user's inputs to the report as well as the parameters.
- For instance, the report requires the user to give a write-up of the model description.
- All of the fields are required in the report.

### `constants.py`
- This file will provide the exact names of the feature columns, target column, a mapping of categorical feature to its one-hot encoded features, and fairness configuration.
- `FEATURES`, `TARGET`, `CATEGORY_MAP` and `CONFIG_FAI` are required in the report.

### `report_utils.py`
- This file is the entry point for the report 
- Write your own `load_model`, `load_data`, `predict`, `compute_shap_values`, `print_model_perf`.
- Model must be a fitted `sklearn` model. If not, write a class to wrap it like a fitted `sklearn` model.
- Sample data must be a `pandas.DataFrame`.
- Feature names and a category map for one-hot encoded features must be furnished in `constants.py`.
- Set the relevant explainer in `compute_shap_values` for your model.
  - `shap.TreeExplainer` works with tree models.
  - `shap.DeepExplainer` works with Deep Learning models.
  - `shap.KernelExplainer` works with all models, though it is slower than other Explainers and it offers an approximation rather than exact Shap values.
- See [Explainers](https://shap.readthedocs.io/en/latest/#explainers) for more details.


### Generate the report
To generate the report, run the following in the command line:

`streamlit run report_utils.py`

You can then view the report in your browser.

To save the report as a PDF, just print the report as PDF.


## Non-exhaustive list of prohibited features
- religion
- nationality
- birth_place
- gender
- race
- education
- neighbourhood
- country/region
