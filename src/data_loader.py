import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df[['Score', 'Summary', 'Text']].dropna()
    df = df[df['Score'] != 3]  # Remove neutral
    df['label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    df['review'] = df['Summary'] + " " + df['Text']
    return df[['review', 'label']]