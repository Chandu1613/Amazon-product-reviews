import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['Score', 'Summary', 'Text']]
    df = df.dropna()
    return df