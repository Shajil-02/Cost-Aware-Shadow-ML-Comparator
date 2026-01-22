import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(raw_path, train_path, val_path, test_path):
    df = pd.read_csv(raw_path)
    
    # Split into (Train + Val) and Test
    tv, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["fraud"])
    # Split (Train + Val) into Train and Val
    train, val = train_test_split(tv, test_size=0.2, random_state=42, stratify=tv["fraud"])

    # Batch save
    for data, path in zip([train, val, test], [train_path, val_path, test_path]):
        data.to_csv(path, index=False)

def load_data(path):
    df = pd.read_csv(path)
    return df.drop("fraud", axis=1), df["fraud"]