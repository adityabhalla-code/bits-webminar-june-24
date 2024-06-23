
import pandas as pd
import numpy as np
import argparse
import os

# Treating 'yr' column as Ordinal categorical variable, assign higher value to 2012
yr_mapping = {2011: 0, 2012: 1}
# Treat 'mnth' column as Ordinal categorical variable, and assign values accordingly
mnth_mapping = {'January': 0, 'February': 1, 'December': 2, 'March': 3, 'November': 4, 'April': 5,
                'October': 6, 'May': 7, 'September': 8, 'June': 9, 'July': 10, 'August': 11}
# Treat 'season' column as Ordinal categorical variable, and assign values accordingly
season_mapping = {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3}
# Map weather situation
weather_mapping = {'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3}
# Map holiday
holiday_mapping = {'Yes': 0, 'No': 1}
# Map workingday
workingday_mapping = {'No': 0, 'Yes': 1}
# Map hour
hour_mapping = {'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8,
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16,
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23}


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
from sklearn.preprocessing import  OneHotEncoder
# build encoder
encoder = OneHotEncoder()

## Function for pre-processing the dataset

def pre_process(dataframe):
    df = dataframe.copy()
    df = get_year_and_month(df)
    df = impute_weekday(df)
    df['weathersit'].fillna('Clear', inplace=True)
    numerical_features , categorical_features , unused_colms = get_feature_categories(df)
    for col in numerical_features:
        df = handle_outliers(df, col)
    df['yr'] = df['yr'].apply(lambda x: yr_mapping[x])
    df['mnth'] = df['mnth'].apply(lambda x: mnth_mapping[x])
    df['season'] = df['season'].apply(lambda x: season_mapping[x])
    df['weathersit'] = df['weathersit'].apply(lambda x: weather_mapping[x])
    df['holiday'] = df['holiday'].apply(lambda x: holiday_mapping[x])
    df['workingday'] = df['workingday'].apply(lambda x: workingday_mapping[x])
    df['hr'] = df['hr'].apply(lambda x: hour_mapping[x])
    encoder.fit(df[['weekday']])
    try:
        enc_wkday_features = encoder.get_feature_names_out(['weekday'])
    except Exception as e:
        print(f"Exception occured on get_features_names_out--{e}")
        enc_wkday_features = [f'weekday_{i}' for i in range(encoder.categories_[0].shape[0])]
    encoded_weekday = encoder.transform(df[['weekday']]).toarray()
    df[enc_wkday_features] = encoded_weekday
    # drop not required columns
    unused_colms.append('weekday')
    df.drop(labels = unused_colms, axis = 1, inplace = True)
    print("preprocessing done !")
    return df




def _parse_args():
    
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='bike-sharing-dataset.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    
    return parser.parse_known_args()


if __name__=="__main__":
    

    # Process arguments
    args, _ = _parse_args()
    
    # Verify if the file exists
    if not os.path.exists(os.path.join(args.filepath, args.filename)):
        raise FileNotFoundError(f"File not found: {input_file_path}")

    
    target_col = "cnt"
    
    # Load data
    df_data = pd.read_csv(os.path.join(args.filepath, args.filename))
    try:
        df_data.drop(['Unnamed: 0'],axis=1 , inplace=True)
    except Exception as e:
        print(f"Exception occured in dropping column--unnamed: 0-->{e}")
    processed_data = pre_process(df_data)

    # Shuffle and splitting dataset
    train_data, validation_data, test_data = np.split(
    processed_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(processed_data)), int(0.9 * len(processed_data))],)

    print(f"Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")

    
    # Save datasets locally
    train_data.to_csv(os.path.join(args.outputpath, 'train/train.csv'), index=False, header=True)
    validation_data.to_csv(os.path.join(args.outputpath, 'validation/validation.csv'), index=False, header=True)
    test_data[target_col].to_csv(os.path.join(args.outputpath, 'test/test_y.csv'), index=False, header=True)
    test_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_x.csv'), index=False, header=True)
    
    # Save the baseline dataset for model monitoring
    processed_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'baseline/baseline.csv'), index=True, header=False)
    
    print("## Processing complete. Exiting.")
