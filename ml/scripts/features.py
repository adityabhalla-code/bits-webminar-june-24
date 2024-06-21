from pathlib import Path
import pandas as pd
import yaml
import sys
import os 

file = Path(__file__).resolve()
parent , root = file.parent , file.parents[1]
sys.path.append(str(root))

current_dir = os.path.dirname(__file__)
config_path = os.path.join(current_dir, '..', 'config.yml')
# import config
with open(config_path) as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

print(config)


# find numerical and categorical variables
def get_feature_categories(df):
    unused_colms = ['dteday', 'casual', 'registered']   # unused columns will be removed at later stage
    target_col = ['cnt']

    numerical_features = []
    categorical_features = []

    for col in df.columns:
        if col not in target_col + unused_colms:
            if df[col].dtypes == 'float64':
                numerical_features.append(col)
            else:
                categorical_features.append(col)
    print('Number of numerical variables: {}'.format(len(numerical_features)),":" , numerical_features)
    print('Number of categorical variables: {}'.format(len(categorical_features)),":" , categorical_features)
    return numerical_features , categorical_features , unused_colms


 # Working on `dteday` column to extract year and month
def get_year_and_month(dataframe):
    df = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()
    print("Added yr and month variables to the dataframe--")
    return df


# Function to impute weekday by extracting day name from the date column

def impute_weekday(dataframe):
    df = dataframe.copy()
    wkday_null_idx = df[df['weekday'].isnull() == True].index
    # print(len(wkday_null_idx))
    df.loc[wkday_null_idx, 'weekday'] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])
    print("imputed weekday in dataframe--")
    return df


# Function to handle outliers for a single column

def handle_outliers(dataframe, colm):
    df = dataframe.copy()
    q1 = df.describe()[colm].loc['25%']
    q3 = df.describe()[colm].loc['75%']
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in df.index:
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound
    print(f"outliers handled in column--{colm}")
    return df


# Treating 'weekday' column as a Nominal categorical variable, perform one-hot encoding
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# build encoder
encoder = OneHotEncoder(sparse_output=False)

## Function for pre-processing the dataset

def pre_process(dataframe):
    df = dataframe.copy()
    df = get_year_and_month(df)
    df = impute_weekday(df)
    df['weathersit'].fillna('Clear', inplace=True)
    numerical_features , categorical_features , unused_colms = get_feature_categories(df)
    for col in numerical_features:
        df = handle_outliers(df, col)
    # print(df.info())
    df['yr'] = df['yr'].apply(lambda x: config['yr_mappings'][x])
    df['mnth'] = df['mnth'].apply(lambda x: config['mnth_mappings'][x])
    df['season'] = df['season'].apply(lambda x: config['season_mappings'][x])
    df['weathersit'] = df['weathersit'].apply(lambda x: config['weathersit_mappings'][x])
    df['holiday'] = df['holiday'].apply(lambda x: config['holiday_mappings'][x])
    df['workingday'] = df['workingday'].apply(lambda x: config['workingday_mappings'][x])
    # df['hr'] = df['hr'].apply(lambda x: config['hour_mappings'][x])
    df['hr'] = df['hr'].replace(config['hr_mappings'])
    encoder.fit(df[['weekday']])
    enc_wkday_features = encoder.get_feature_names_out(['weekday'])
    encoded_weekday = encoder.transform(df[['weekday']])
    df[enc_wkday_features] = encoded_weekday
    # drop not required columns
    unused_colms.append('weekday')
    df.drop(labels = unused_colms, axis = 1, inplace = True)
    print("preprocessing done !")
    return df
