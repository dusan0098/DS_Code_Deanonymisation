from typing import Dict
from typing import Iterable
from typing import List

from javalang.tokenizer import JavaToken, Identifier, Keyword, Literal
from javalang.tree import Node

import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression

# returns dataframe from csv and
def dataframe_from_csv(suffix = ".java", year = "2009"):
    chunksize = 1000
    df = pd.DataFrame()
    
    file_name = f"data/gcj{year}.csv/gcj{year}.csv"
    with pd.read_csv(file_name, chunksize=chunksize) as reader:
        for chunk in reader:
            new_df = chunk[chunk['file'].str.endswith(suffix)]
            df = pd.concat([df,new_df])
    
    return df

# preparing dataframe
def prepare_dataframe(df):
    # take important columns and rename
    username_to_id = map_user_to_ids(df['username'])
    user_ids = []

    for idx, element in df.iterrows():
        user_ids.append(username_to_id[element["username"]])
    df['user_id'] = user_ids

    df = df[[ 'user_id', 'username', 'round', 'task', 'full_path', 'flines', 'solution']]
    df = df.rename(columns = {'task':'problem_id','round':'round_id','full_path':'path', 'flines':'code'})
    
    # returns a list of all rows that don't have code
    list_empty_errors = []
    for idx, row in df.iterrows():
        if (len(str(row["code"])) < 10):
            list_empty_errors.append(idx)

    # drop all the rows with no code
    df_drop = df.drop(list_empty_errors).reset_index(drop=True)
    
    return df_drop

# returns a sample from dataframe
def sample_from_dataframe(df, num_files, num_users):
    #select users with more than n_files
    count = df.groupby('user_id').problem_id.count()
    users = count[count >= num_files].index
    users = np.random.choice(users, num_users, replace=False)
    
    #select n_files from all n_users
    parts = [df[df.user_id == user].sample(n=num_files, replace=False) for user in users]
    dataset = pd.concat(parts).reset_index(drop=True)

    # Create new user ids
    user_id_to_new_id = map_user_to_ids(dataset.user_id)
    dataset.user_id = dataset.user_id.apply(lambda it: user_id_to_new_id[it])
    return dataset

# select best k features according to MI
def feature_selection(dataset, num_features):
    
    num_features = np.min([num_features, dataset.shape[1]-1])
    print("Selecting best features...")
    start_time = datetime.now()

    X, y = dataset.loc[:, dataset.columns != "user_id"], dataset.user_id.values
    
    mi = mutual_info_regression(np.nan_to_num(X), y, random_state=0)
    mi /= np.max(mi)
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    print(f"Feature selection lasted: {time_elapsed.total_seconds()}")
    
    # sort and take best n features according to MI
    mi_index = np.argsort(mi)
    features_index = mi_index[-num_features:]
    features = X.columns[features_index].values
    X = X[features]
    
    # Check if any are Na
    nan_count = X.isna().sum(axis=0)
    indices = np.argsort(nan_count.values)
    features = nan_count[indices][:num_features].index
    X = X[features]
    print(f"Number of features selected: {X.shape[1]}")
    
    return {"X":X, "y":y, "mi":np.sort(mi)[-num_features:]}

# return hard problems from dataframe
def get_hard_problems(df):
    
    count_solution = df.groupby('problem_id').solution.count()
    sum_solution = df.groupby('problem_id').solution.sum()

    solution_df = pd.DataFrame([count_solution,sum_solution], index=["count","solution"])
    solution_df = solution_df.T
    solution_df.sort_values("solution", ascending=False)

    hard_problems = list(solution_df[solution_df["solution"] < 30].index)
    
    return hard_problems

def get_best_users(df, hard_problems):
    
    df_best = df[df['problem_id'].isin(hard_problems)]
    df_best = df_best[df_best['solution'].isin([1])]
    
    best_users = list(set(df_best['username']))

    return best_users

# returns list of Javalang identifier
def identifiers(tokens):
    return [x for x in tokens if isinstance(x, Identifier)]

# returns list of Javalang keywords
def keywords(tokens):
    return [x for x in tokens if isinstance(x, Keyword)]

# returns list of Javalang literals
def literals(tokens):
    return [x for x in tokens if isinstance(x, Literal)]

# returns children of a node
def children(node):
    nodes = []

    for child in node.children:
        if isinstance(child, List):
             nodes += child
        else:
            nodes.append(child)

    return nodes

# returns non empty lines
def non_empty_lines(code):
    return [line for line in code.split('\n') if line.strip() != '']


# return all nodes of a given type starting from node as root
def get_nodes(node, node_type):
    result = []

    if isinstance(node, node_type):
        result.append(node)

    for it in children(node):
        if isinstance(it, Node):
            result += get_nodes(it, node_type)

    return result

# returns count of all nodes of a type from root node
def get_nodes_count(node, node_type):
    return len(get_nodes(node, node_type))

# returns ids from users
def map_user_to_ids(users):
    users = sorted(set(users))
    return {key: value for key, value in zip(users, range(len(users)))}