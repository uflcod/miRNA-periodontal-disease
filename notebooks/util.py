import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Union, Tuple, Dict, List, Any
import xgboost as xgb
import shap
from sklearn.model_selection \
    import RandomizedSearchCV, GridSearchCV, LeaveOneOut, cross_val_score
from pprint import pprint

# supress sklearn warnings (hopefully)
import warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def make_study_df(
        miRNA_df:pd.DataFrame, 
        infected_str:str='tf_', 
        cohort_str:Union[None, str]=None,
        shorten_miRNA_name:bool=True
    ) -> pd.DataFrame:

    df = miRNA_df.copy()

    # subset to cohort
    if cohort_str:
        df = df[df.cohort.str.endswith(cohort_str)]

    # short the col name of the miRNA
    # e.g., mmu-miR-1971 (MIMAT0009446) -> miR-1971
    if shorten_miRNA_name:
        # the first split get everything after the first "-" (e.g., removes "mmu-")
        # the second split gets everthing before the MIMAT number
        # note, we don't want to modify the "cohort" column
        cols = [
            "cohort" if col == "cohort" else "-".join(col.split("-")[1:]).split(" ")[0]
            for col in df.columns
        ]

        df.columns = cols # re-assign columns

    # add column to identify infected mice
    df['infected'] = np.where(df.cohort.str.startswith(infected_str), 1, 0)

    # drop cohort, number, and name data
    df = df.iloc[:, 4:] 

    # shuffle data
    df = df.sample(frac=1, random_state=1989)

    # shuffle columns
    cols = list(df.columns)
    random.shuffle(cols)
    df = df[cols]
    
    return df


def param_search_cv(
        param_grid:Dict, 
        X_data:pd.DataFrame,
        y_targets: Union[pd.Series, List, np.ndarray],
        model:Any,
        cvs:List,
        search_type:str='random',
        n_iter:int=100,
        random_state:int=42,
        n_jobs:int=-1,
        print_iter:bool=False,
        print_scores:bool=False,
        print_best:bool=False
) -> Dict:
    scores = []
    best_search = {
        'score': 0,
        'cv': 0,
        'params': {}
    }
    for cv in cvs:
        params = \
            param_search(
                param_grid, 
                X_data, 
                y_targets, 
                model, 
                cv=cv,
                search_type=search_type,
                n_iter=n_iter,
                random_state=random_state,
                n_jobs=n_jobs, 
                print_best=False)
        m = model(**params).fit(X_data, y_targets)
        sc = np.mean(cross_val_score(m, X_data, y_targets, cv=LeaveOneOut()))
        scores.append(sc)
        if sc > best_search['score']:
            best_search['score'] = sc
            best_search['cv'] = cv
            best_search['params'] = params

        if print_iter:
            print(f'cv: {cv}', f'score: {sc}')

    if print_scores:
        pprint(scores)

    if print_best:
        pprint(best_search)

    return best_search


def param_search(
        param_grid:Dict, 
        X_data:pd.DataFrame,
        y_targets: Union[pd.Series, List, np.ndarray],
        model:Any,
        search_type:str='random',
        n_iter:int=100,
        cv:int=3,
        random_state:int=42,
        n_jobs=-1,
        print_best=True
) -> Dict:
    if 'random' == search_type:
        param_model = RandomizedSearchCV(
            estimator=model(),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif 'grid' == search_type:
        param_model = GridSearchCV(
            estimator=model(),
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs
        )
    else:
        raise ValueError("search_type must be 'random' or 'grid'")
    
    param_model.fit(X_data, y_targets)
    if True == print_best:
        pprint(param_model.best_params_)
    return param_model.best_params_

def plot_xgb_feature_importance(
    model: xgb.XGBModel, 
    title:str='',
    num_features:int=5,
    size_width:int=8,
    size_height:int=4,
    font_size:int=12,
    y_label:str='',
    save_fig:bool=False,
    dpi:int=300,
    figures_dir:str='../figures/',
    file_name:str='',
    show_plot:bool=True
):
    fig, ax = plt.subplots()

    xgb.plot_importance(model, max_num_features=num_features, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)
    plt.ylabel(y_label)
    plt.title(title)
    plt.gcf().set_size_inches(size_width, size_height)
    
    if save_fig:
        if len(file_name) > 0:
                fname = f'{figures_dir}{file_name}'
        elif len(title) > 0:
            fname = f'{figures_dir}{title.replace(" ", "_")}.png'
        else:
            fname = f'{figures_dir}xgboost_feature_importance_{random.randint(1, 1_000_000)}.png'
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    
    if show_plot:
        plt.show()


# see https://stackoverflow.com/questions/39803385/what-does-a-4-element-tuple-argument-for-bbox-to-anchor-mean-in-matplotlib
# for explanation of bbox_to_anchor values
def plot_shap_feature_importance(
    shap_values:Any, 
    title:str='',
    num_features:int=6,
    size_width:int=8,
    size_height:int=4,
    font_size:int=12,
    y_label:str='',
    cohorts:int=2,
    legend_loc:str='best',
    bbox_values:Tuple=(0, 0, 1, 1),
    save_fig:bool=True,
    dpi:int=300,
    figures_dir:str='../figures/',
    file_name:str='',
    show_plot:bool=True
):

    if cohorts > 0:
        shap.plots.bar(shap_values.cohorts(cohorts).abs.mean(0), max_display=num_features, show=False)
    else:
        shap.plots.bar(shap_values, max_display=num_features, show=False)

    ax = plt.gca()
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)
    plt.ylabel(y_label)
    plt.title(title)
    if cohorts > 0: 
        plt.legend(loc=legend_loc, bbox_to_anchor=bbox_values)
    plt.gcf().set_size_inches(size_width, size_height)

    if save_fig:
        if len(file_name) > 0:
                fname = f'{figures_dir}{file_name}'
        elif len(title) > 0:
            fname = f'{figures_dir}{title.replace(" ", "_")}.png'
        else:
            fname = f'{figures_dir}shap_values_feature_importance_{random.randint(1, 1_000_000)}.png'
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    
    if show_plot:
        plt.show()


def plot_shap_summary(
    shap_values:Any,
    X_data:pd.DataFrame,
    title:str='',
    plot_type:str='dot',
    num_features:int=5,
    size_width:int=6,
    size_height:int=3,
    font_size:int=12,
    font_name:str='Times New Roman',
    font_weight:str='bold',
    y_label:str='',
    save_fig:bool=False,
    dpi=300,
    figures_dir:str='../figures/',
    file_name:str='',
    show_plot:bool=True
):
    fontparams = {'fontname':font_name, 'fontsize': font_size, 'weight': font_weight}

    shap.summary_plot(shap_values, X_data, max_display=num_features, show=False, plot_type=plot_type)
    ax = plt.gca()
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)
    ax.set_yticklabels(ax.get_yticklabels(),**fontparams)
    plt.ylabel(y_label, **fontparams)
    plt.title(title, **fontparams)
    plt.gcf().set_size_inches(size_width, size_height)

    if save_fig:
        if len(file_name) > 0:
                fname = f'{figures_dir}{file_name}'
        elif len(title) > 0:
            fname = f'{figures_dir}{title.replace(" ", "_")}.png'
        else:
            fname = f'{figures_dir}shap_values_summary_{random.randint(1, 1_000_000)}.png'
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    
    if show_plot:
        plt.show()


def plot_shap_heatmap(
    shap_values:Any,
    title:str='',
    x_label:str='',
    supxlabel:str='',
    num_features:int=5,
    size_width:int=8,
    size_height:int=5,
    font_size:int=12,
    font_name:str='Times New Roman',
    font_weight:str='bold',
    save_fig:bool=False,
    dpi=300,
    figures_dir:str='../figures/',
    file_name:str='',
    show_plot:bool=True
):
    fontparams = {'fontname':font_name, 'fontsize': font_size, 'weight': font_weight}

    shap.plots.heatmap(shap_values, max_display=num_features, show=False)
    ax = plt.gca()
    ax.set_yticklabels(ax.get_yticklabels(), **fontparams)
    
    plt.title(title, **fontparams)
    
    if len(x_label) > 0:
        ax.set_xlabel(x_label, **fontparams)

    if len(supxlabel) > 0:
        plt.gcf().supxlabel(supxlabel, **fontparams)
    plt.gcf().set_size_inches(size_width, size_height)

    if save_fig:
        if len(file_name) > 0:
                fname = f'{figures_dir}{file_name}'
        elif len(title) > 0:
            fname = f'{figures_dir}{title.replace(" ", "_")}.png'
        else:
            fname = f'{figures_dir}shap_values_summary_{random.randint(1, 1_000_000)}.png'
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')

    if show_plot:   
        plt.show()

    return plt.gcf()


def plot_shap_dependence(
    shap_values:Any,
    X_data:pd.DataFrame,
    title:str='',
    size_width:int=8,
    size_height:int=5,
    font_size:int=12,
    font_name:str='Times New Roman',
    font_weight:str='bold',
    save_fig:bool=False,
    dpi:int=300,
    figures_dir:str='../figures/',
    file_name:str='',
    show_plot:bool=True
):
    fontparams = {'fontname':font_name, 'fontsize': font_size, 'weight': font_weight}

    shap.dependence_plot("rank(0)", shap_values.values, X_data, display_features=X_data, show=False)
    ax = plt.gca()
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)
    ax.set_xlabel(ax.get_xlabel(), **fontparams)
    ax.set_ylabel(ax.get_ylabel().replace('\n', ' '), **fontparams)
    plt.title(title, **fontparams)
    plt.gcf().set_size_inches(size_width, size_height)

    if save_fig:
        if len(file_name) > 0:
                fname = f'{figures_dir}{file_name}'
        elif len(title) > 0:
            fname = f'{figures_dir}{title.replace(" ", "_")}.png'
        else:
            fname = f'{figures_dir}shap_dependence_plot_{random.randint(1, 1_000_000)}.png'
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    
    if show_plot:
        plt.show()


def plot_shap_importance_with_summary(
    shap_values:Any,
    X_data:pd.DataFrame,
    title:str='',
    supxlabel:str='',
    num_features:int=5,
    size_width:int=8,
    size_height:int=4,
    font_size:int=12,
    font_name:str='Times New Roman',
    font_weight:str='bold',
    y_label:str='',
    cohorts:int=2,
    legend_loc:str='best',
    bbox_values:Tuple=(0, 0, 1, 1),
    save_fig:bool=False,
    dpi:int=300,
    figures_dir:str='../figures/',
    file_name=''
):
    fontparams = {'fontname':font_name, 'fontsize': font_size, 'weight': font_weight}

    fig = plt.figure()

    ax1 = plt.subplot(121)
    plot_shap_summary(
        shap_values,
        X_data,
        title='',
        plot_type='bar',
        num_features=num_features,
        size_width=size_width,
        size_height=size_height,
        font_size=font_size,
        font_name=font_name,
        font_weight=font_weight,
        y_label=y_label,
        save_fig=False,
        dpi=dpi,
        figures_dir=figures_dir,
        show_plot=False
    )
    ax1.set_xlabel('')

    ax2 = plt.subplot(122)
    
    plot_shap_summary(
        shap_values,
        X_data,
        title='',
        num_features=num_features,
        size_width=size_width,
        size_height=size_height,
        font_size=font_size,
        y_label=y_label,
        save_fig=False,
        dpi=dpi,
        figures_dir=figures_dir,
        show_plot=False
    )
    ax2.set_yticklabels('')
    ax2.set_ylabel('')
    ax2.set_xlabel('')

    if len(title) > 0:
        fig.suptitle(title)
    
    # set label at bottom figure
    fig.supxlabel(supxlabel, **fontparams)
    fig.tight_layout()

    # plt.show() # this keeps the figure from saving ... I don't know why.
    
    if save_fig:
        if save_fig:
            if len(file_name) > 0:
                fname = f'{figures_dir}{file_name}'
            elif len(title) > 0:
                fname = f'{figures_dir}{title.replace(" ", "_")}_with_summary.png'
            else:
                fname = f'{figures_dir}shap_importance_with_summary_plot_{random.randint(1, 1_000_000)}.png'
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')


# NOT WORKING RIGHT NOW!!!
def combine_shap_heatmaps(
        heatmap1_params:Dict,
        heatmap2_params:Dict,
        heatmap3_params:Dict,
        title:str='',
        dpi:int=300,
        figures_dir:str='../figures/',
        show_plot:bool=True,
        save_fig=True
    ):
    fig = plt.figure()
    
    ax1 = plt.subplot(121)
    plot_shap_heatmap(**heatmap1_params)
    # fig.colorbar()
    # clb = plt.gcf().colorbar()
    # clb.ax.set_titile('')

    ax2 = plt.subplot(122)
    plot_shap_heatmap(**heatmap2_params)

    # ax3 = plt.subplot(211)
    # plot_shap_heatmap(**heatmap3_params)

    if len(title) > 0:
        fig.suptitle(title)
    
    # fig.supxlabel('Instances')
    fig.tight_layout()

    if show_plot:
        plt.show()
    
    if save_fig:
        if save_fig:
            if len(title) > 0:
                fname = f'{figures_dir}{title.replace(" ", "_")}_combined_heatmaps.png'
            else:
                fname = f'{figures_dir}shap_combined_heatmaps_plot_{random.randint(1, 1_000_000)}.png'
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')


def top_shap_values(shap_values:Any, top_n:int=5, print_importance:bool=False):
    feature_names = shap_values.feature_names # get column/feature names
    vals = np.abs(shap_values.values).mean(axis=0) # calc the mean shap vals

    # build data frame and sort
    shap_importance = pd.DataFrame(zip(feature_names, vals), columns=['feature', 'shap_val'])
    shap_importance.sort_values(by=['shap_val'], ascending=False, inplace=True)
    if top_n > 0:
        shap_importance = shap_importance.head(top_n)
    
    if print_importance:
        print(shap_importance)

    # return shap values using indexes from sorted dataframe
    idx = list(shap_importance.index)
    ret_val = shap_values[:, idx]
    ret_val.feature_names = list(shap_importance['feature'])
    return ret_val