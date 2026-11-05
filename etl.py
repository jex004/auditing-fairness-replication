from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

def load_data():
    adult = fetch_ucirepo(id=2)
    return adult.data.features.join(adult.data.targets)

def clean_df(df):

    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    df = df.replace('?', np.nan)
    df = df.fillna('Unknown')
    
    #make features binary 
    df['US-native'] = df['native-country'].apply(lambda x: 1 if 'United-States' in str(x) else 0)
    df['above-50k-income'] = df['income'].str.rstrip('.').apply(lambda x: 1 if '>50k' in str(x).lower() else 0)
    df['male-gender'] = df['sex'].apply(lambda x: 1 if 'Male' in str(x) else 0)

    #binning education into proper grade levels
    elementary = [1, 2]
    middle = [3, 4]
    hs = [5, 6, 7, 8]
    hs_grad = [9, 10]
    associate = [11, 12]
    bachelors = [13]
    masters = [14, 15]
    doc = [16]
    
    bin_labels = []
    for num in df['education-num']:
        if num in elementary:
            bin_labels.append('Elementary')
        elif num in middle:
            bin_labels.append('Middle')
        elif num in hs:
            bin_labels.append('HS Partial')
        elif num in hs_grad:
            bin_labels.append('HS Grad')
        elif num in associate:
            bin_labels.append('Associate')
        elif num in bachelors:
            bin_labels.append('Bachelors')
        elif num in masters:
            bin_labels.append('Masters')
        elif num in doc:
            bin_labels.append('Doctorate')
            
    df['education'] = bin_labels

    #binning workclass into proper categories
    workclass_bins = {
    'Private': 'Private',
    'Federal-gov': 'Gov',
    'Local-gov': 'Gov',
    'State-gov': 'Gov',
    'Self-emp-not-inc': 'Self-Employed',
    'Self-emp-inc': 'Self-Employed',
    'Without-pay': 'Unemployed',
    'Never-worked': 'Unemployed',
    'Unknown': 'Unknown'
    }
    
    df['workclass'] = df['workclass'].map(workclass_bins).fillna('Unknown')
        
    df = df.drop(columns=['sex','fnlwgt', 'native-country', 'income', 'education-num'])
    
    # just reordering columns
    df = df[['age',
             'male-gender',
             'US-native',
             'race',
             'education',
             'workclass',
             'occupation',
             'hours-per-week', 
             'marital-status',  
             'relationship', 
             'capital-gain', 
             'capital-loss', 
             'above-50k-income']]
    
    return df