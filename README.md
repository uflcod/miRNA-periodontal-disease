# miRNA periodontal disease

## Purpose
This repository is for developing methods to analyze the assocations between miRNAs and periodontal disease.

## Notebooks 
The repository contains the following notebooks in the `notebooks` directory.

### AI Notebooks 
- `xgboost_miRNA.ipynb`  
  Runs XGBoostClassifer on three datasets:
  - Sham and infeceted mice from all weeks; i.e., 8 week and 16 week datasets are merged.
  - Sham and infected mice at 8 weeks.
  - Sham and infected mice at 16 weeks.
  
  In each dataset, there is a flag (named ‘infected’) that marks whether the mice came from the infected group or the sham group.  
  XGBoost's variable importance and SHAP values are then used to determine which miRNA variable was most important in each cohort dataset. 

### Utility Notebooks
- `transpose_merge_miRNA.ipynb`  
  Run notebook to transpose and merge the miRNA data files into a single `transposed_Tf_miRNA.xlsx` file.
- `transpose_Tf_aveolar_bone_resporption.ipynb`  
  Run notebook to transpose the `Tf_aveolar_bone_resporption.xlsx` data into the `transposed_Tf_aveolar_bone_resporption.xlsx` file.
- `merge_miRNA_bone_resorption.ipynb `  
   Run notebook to create the `merged_miRNA_resporption.xlsx` data file.

## Virtual environment setup
- To create a virtual environment run the command `python -m venv venv`. 
  This will create a virtual environment in the `venv` directory. You can specify a different directory name if you wish.
- Run `source venv/bin/activate` to activate (i.e., run) the virtual environment.
- **After** activating the environment, install the project libraries using the command `pip install -r requirements.txt -U`.

## Jupyter hack for virtual environments
Jupyter notebooks won't necessarily have access to the libraries installed in the virtual environment.  
One hack to get around this is to create a softlink to the Jupyter binary created in the virtual environment like so `ln -s venv/bin/jupyter`.

You can then start Jupyter within the virtual environment using the softlink. E.g., `./jupyter lab`.  
The allows the Jupyter notebook to access the libraries in the virtual environment.

Another option is create a Jupyter kernel from the virtual environment. See [here](https://towardsdatascience.com/link-your-virtual-environment-to-jupyter-with-kernels-a69bc61728df) for details.
