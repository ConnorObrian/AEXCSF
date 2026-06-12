"""
Custom utility functions for evaluation.
"""
__author__ = "Connor Schönberner"

from collections import namedtuple
from pathlib import Path

import glob
import pandas as pd


Col = namedtuple('Col',['name','first_500', 'overall', 'last_500'])
Datapoint = namedtuple('Datapoint', ['mean', 'std'])

def load_csv(save_path: Path|str, learn_runs: int=1) -> tuple[list[pd.DataFrame], str]:
    """Loads the csv files created during an experimental series (a series of experiments/runs).
       It relies on the name schema that are used for the files. 
       Prefix Experimental for learning (learn_runs == 1), End_Evaluation for end eval runs (learn_runs < 0), 
       Testing for testing (learn_runs == 0).

    Args:
        save_path (Path | str): Where are the csv's stored?
        learn_runs (int, optional): Which type of runs should be evaluated? Defaults to 1 (Learning).

    Returns:
        tuple[list[pd.DataFrame], str]: The loaded csv files as DataFrames 
            and the result prefix string indicating which type of runs has been loaded
    """
    glob_string = str(save_path) 
    
    if learn_runs:
        glob_string += "/Experimental*.csv"
        res_prefix = "Learn"
    elif learn_runs < 0:
        glob_string += "/End_Evaluation_*.csv"
        res_prefix = "End"
    else:
        glob_string += "/Testing*.csv"
        res_prefix = "Test"
    filenames_unsorted = glob.glob(glob_string)
    filenames = sorted(filenames_unsorted)    

    dfs: pd.DataFrame = []
    for f in filenames:
        df = pd.read_csv(f, sep=",", index_col=0)
        
        # if "index" in df.columns:
        #     # Drops the column that records the i-th step at which the episodes ended
        #     df = df.drop("index", axis=1)
        dfs.append(df)
    
    return dfs, res_prefix, filenames

def concat_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the mean and the std of all columns in a dataframe and concatenates them to one dataframe.
       Rounds them to 3 decimals.
       
    Args:
        df (pd.DataFrame): Target DataFrame

    Returns:
        pd.DataFrame: Resulting DataFrame
    """
    return pd.concat([df.mean().round(3), df.std().round(3).add_suffix("_STD")]).sort_index()

def evaluate_dataframes(dfs, filenames, res_prefix="", target_path: Path|str="", sep=",", store=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                  pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shortened and improved evaluation function for multi-run experimental series.
       Stores the results of the evaluation in a directory res_prefix_eval as long as store==True.

    Args:
        target_path (_type_): _description_
        res_prefix (bool, optional): _description_. Defaults to "".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        The means of each of the N runs in DataFrame, -- over the last 500 episodes,-- over the last 100 episodes,
        
    """
        
    # Calculates the means for each single Dataframe/Run     
    means_dfs = []
    means_last_dfs = []
    first_25_percent_dfs = []
    last_25_percent_dfs = []
    rows_25percent: int = dfs[0].shape[0] // 4
    
    for df in dfs:
        means_dfs.append(df.mean(axis=0, numeric_only=True))
        means_last_dfs.append(df.tail(100).mean(axis=0, numeric_only=True))
        last_25_percent_dfs.append(df.tail(rows_25percent).mean(axis=0, numeric_only=True))
        first_25_percent_dfs.append(df.head(rows_25percent).mean(axis=0, numeric_only=True))
   
    # Concatenates them to a DataFrame containing the mean for each single run
    means_of_the_runs = pd.DataFrame(means_dfs)
    last_episodes_means_of_the_runs = pd.DataFrame(means_last_dfs).add_suffix("Last100")
    last_25_percent_means_of_the_runs = pd.DataFrame(last_25_percent_dfs).add_suffix(f"Last{rows_25percent}")
    first_25_percent_means_of_the_runs = pd.DataFrame(first_25_percent_dfs).add_suffix(f"First{rows_25percent}")

    
    # Computes the mean of the run means and the standard deviation between runs
    total_means_stds = concat_mean_std(means_of_the_runs)
    last_means_stds = concat_mean_std(last_episodes_means_of_the_runs)
    last_25_percent_means_stds = concat_mean_std(last_25_percent_means_of_the_runs)
    first_25_percent_means_stds = concat_mean_std(first_25_percent_means_of_the_runs)
        
    # Calculates the mean and standard deviation for each col per episode (basically for each cell) for being able to plot an averaged timeseries over all N runs
    df_concat: pd.DataFrame = pd.concat(dfs)
    meaned_per_index_experimental_series = df_concat.groupby(level=0).mean(numeric_only=True)
    std_per_index_experimental_series = df_concat.groupby(level=0).std(numeric_only=True).add_suffix("_STD")
    meaned_per_index = pd.concat([meaned_per_index_experimental_series, std_per_index_experimental_series], axis=1).sort_index(axis=1)
    
    if store:
        path = target_path / f"{res_prefix}_eval"
        path.mkdir(exist_ok=True)
        total_means = pd.concat([total_means_stds, last_means_stds, last_25_percent_means_stds, first_25_percent_means_stds]).sort_index()
    
        pd.DataFrame(filenames).to_csv(path / "filenames_order.csv", sep=sep)
        total_means.to_csv(path / "total_means.csv", sep=sep)
        means_of_the_runs.to_csv(path / "means_of_the_runs.csv", sep=sep)
        last_episodes_means_of_the_runs.to_csv(path / "last_100_episodes_means_of_the_runs.csv", sep=sep)
        last_25_percent_means_of_the_runs.to_csv(path / "last_25_percent_means_of_the_runs.csv", sep=sep)
        first_25_percent_means_of_the_runs.to_csv(path / "first_25_percent_means_of_the_runs.csv", sep=sep)
        meaned_per_index.round(decimals=3).to_csv(path / "meaned_timeseries.csv", sep=sep)        
        
    return total_means_stds, last_means_stds, last_25_percent_means_stds, first_25_percent_means_stds, means_of_the_runs, last_episodes_means_of_the_runs, meaned_per_index

def eval_and_store_experimental_series(path, test=False, end=False):
    """Helper function to load all csvs of the runtype and to evaluate them.

    Args:
        path (_type_): target_path
        test (bool, optional): Also evaluate test runs. Defaults to False.
        end (bool, optional): Also evaluate end evaluation runs. Defaults to False.
    """
    l = [1]
    
    if test:
        l.append(0)
    
    if end:
        l.append(-1)
    
    for run_type in l:
        dfs, res_prefix, filenames = load_csv(path, learn_runs=run_type)
        evaluate_dataframes(dfs, filenames=filenames, target_path=Path(path), res_prefix=res_prefix, sep=",")
