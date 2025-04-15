import pandas as pd
import torch
import torch.nn.functional as F

def convert_cat_to_one_hot(df, cat_col):
    # Sample categorical data
    categories = df[cat_col]

    # Create a mapping from category to index
    category_to_index = {category: index for index, category in enumerate(sorted(list(set(categories))))}

    # Convert categories to indices
    category_indices = [category_to_index[category] for category in categories]

    # One-hot encode the indices
    num_categories = len(category_to_index)
    one_hot_encoded = F.one_hot(torch.tensor(category_indices), num_classes=num_categories).float().numpy()
    rdf=pd.DataFrame(one_hot_encoded, columns=[f'{cat_col}_{x}' for x in category_to_index.keys()])

    return num_categories, category_to_index, rdf

def process_cat_cols(df, cat_cols):
    res_df=pd.DataFrame()
    cat_index_dict={}
    for cat_col in cat_cols:
        num_categories, category_to_index, rdf=convert_cat_to_one_hot(df, cat_col)
        res_df=pd.concat([res_df, rdf], axis=1)
        cat_index_dict[cat_col]=category_to_index

    return res_df, cat_index_dict