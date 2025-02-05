import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from constant import *

def extract_elements(s):
    matches = re.findall(r"-?\d+", s)
    digits = [int (match) for match in matches]
    letter = re.findall(r'[a-zA-Z]+', s)
    return digits + letter


def return_exist_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Check if columns exist in the DataFrame and return the columns that exist.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to check columns in.
    columns (List[str]): The list of column names to check for existence.
    
    Returns:
    List[str]: A list of column names that exist in the DataFrame.
    """
    return [col for col in columns if col in df.columns]

# convert missing or unknown data to numpy NaNs.
def convert_missing_value_to_NaNs(x, missing_value_list):
    if x in missing_value_list:
        return np.nan
    else:
        return x

def compare_column_distribution(df1, df2, column):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sns.countplot(x=column, data=df1, ax=axes[0], hue=column, palette="Blues", legend=False)
    axes[0].set_title(f'Distribution in Lower Missing Data Group')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Count")

    sns.countplot(x=column, data=df2, ax=axes[1], hue=column, palette="Oranges", legend=False)
    axes[1].set_title(f'Distribution in Higher Missing Data Group')
    axes[1].set_xlabel(column)
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()
    
    
def clean_data(df: pd.DataFrame, feature_missin_values, imp_mean):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    for feature_name, missing_value_list in feature_missin_values.items():
        df[feature_name] = df[feature_name].replace(missing_value_list, np.nan)
    # remove selected columns and rows, ...
    
    ## columns 
    missing_data_proportion = df.isna().mean()
    customer_columns_to_drop = missing_data_proportion[missing_data_proportion > 0.4]
    df.drop(customer_columns_to_drop.index, axis=1, inplace=True)

    plt.figure(figsize=(10, 6))
    missing_data_proportion.hist(bins=80, edgecolor='black')
    plt.title("Distribution of Missing Value Proportions")
    plt.xlabel("Proportion of Missing Values")
    plt.ylabel("Number of Features")
    plt.show()
    
    ## rows
    df = df[df.isna().mean(axis=1) <= 0.3]
    
    # select, re-encode, and engineer column values.
    feature_need_reEncoding = ['ANREDE_KZ', 'VERS_TYP', 'OST_WEST_KZ', 'NATIONALITAET_KZ', 'SHOPPER_TYP', 'GEBAEUDETYP', 'CAMEO_DEUG_2015', 'GFK_URLAUBERTYP', 'CAMEO_DEU_2015']
    df = pd.get_dummies(df, columns=feature_need_reEncoding)
    
    #PRAEGENDE_JUGENDJAHRE
    df['PRAEGENDE_JUGENDJAHRE_decade'] = df['PRAEGENDE_JUGENDJAHRE'].map(decade_map)
    df['PRAEGENDE_JUGENDJAHRE_movement'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement_map)
    df.drop('PRAEGENDE_JUGENDJAHRE', axis=1,inplace=True)
    
    df['CAMEO_INTL_2015_wealth'] = pd.to_numeric(df['CAMEO_INTL_2015'].str[0])
    df['CAMEO_INTL_2015_lifeStage'] = pd.to_numeric(df['CAMEO_INTL_2015'].str[1])
    df.drop('CAMEO_INTL_2015', axis=1,inplace=True)
    
    #
    bool_columns = df.select_dtypes(include='bool').columns
    for col in bool_columns:
        df[col] = df[col].astype(int)

    ## filling missing data
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.set_output(transform='pandas')
    df = imp_mean.fit_transform(df)
    
    # Return the cleaned dataframe.
    
    return df