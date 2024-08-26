import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(data_name, data_path, sample_size, seed=42):
    '''
    Load data
    Args:
        data_name: dataset name (choices: 'adult', 'bank')
        data_path: path storing dataset
        sample_size: using sample size. if sample_size < 0, then all data is used. otherwise, some samples are sampled ramdomly
        seed: random seed
    Returns:
        data_df: pd.DataFrame
        cat_cols: list of categorical feature names
    '''
    
    if data_name == "adult":
        data_df = pd.read_csv(os.path.join(data_path, data_name, "adult.data"), header=None)
        data_df.rename(columns={0:"age", 1:"workclass", 2:"fnlwgt", 3:"education", 4:"education-num",
                                5:"marital-status", 6:"occupation", 7:"relationship", 8:"race", 9:"sex",
                                10:"capital-gain", 11:"capital-loss", 12:"hours-per-week", 13:"native-country",
                                14:"target"},
                        inplace=True)
        cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]

        # delete rows that have missing values
        data_df = data_df.dropna(how='any')
        # label encoding
        le = LabelEncoder()
        le_cols = ['sex', 'target']
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
    
    elif data_name == "bank":
        data_df = pd.read_csv(os.path.join(data_path, data_name, "bank-full.csv"), sep=';')
        data_df = data_df.rename(columns={"y":"target"})
        data_df = data_df.drop_duplicates().reset_index(drop=True) # 重複データの削除

        def month2num(x):
            return str(x).replace('jan','1').replace('feb', '2').replace('mar', '3').replace('apr', '4').replace('may','5').replace('jun', '6').replace('jul', '7').replace('aug', '8').replace('sep', '9').replace('oct','10').replace('nov', '11').replace('dec', '12')

        data_df['month'] = data_df['month'].map(month2num).astype(int)

        cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
        le_cols = ['default', 'housing', 'loan', 'target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])
        
        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    else:
        raise NotImplementedError
    
    if sample_size > 0:
        data_df = data_df.sample(n=sample_size, random_state=seed)
        data_df.reset_index(drop=True, inplace=True)

    return data_df, cat_cols

# for experiments' file name
def comp_cols_code(dataset_name, comp_cols):
    if dataset_name == 'bank':
        cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
    elif dataset_name == 'adult':
        cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]
    else:
        raise NotImplementedError
    
    code = ""
    for cat in cat_cols:
        if cat in comp_cols:
            code += "1"
        else:
            code += "0"
            
    code = int(code, 2)
    
    return code