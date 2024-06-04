import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_starting_data():
    data = pd.read_csv('winequality-red.csv', sep=";")
    return data

def display_histogram(data):

    data.hist(bins=50, figsize=(20,15))
    plt.show()

def display_info_data(data):
    print(data.info())


def modify_data_initial(data):
    # Remove non-ASCII characters
    for column in data.columns:
        data[column] = data[column].astype(str).str.replace(r'[^\x00-\x7F]+', '', regex=True)
    
    # Change types of quantitative variables
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Drop rows with any NaN values
    data.dropna(inplace=True)

    # Filter outliers
    for column in data.columns:
        if column != 'quality' and pd.api.types.is_numeric_dtype(data[column]):
            Q1 = data[column].quantile(0.10)
            Q3 = data[column].quantile(0.90)
            IQR = Q3 - Q1

            filter = (data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)
            data = data.loc[filter]
    return data

def corr_matrix(data):
    plt.figure(figsize = (20, 18)) 
    # creating the correlation heatmap
    sns.heatmap(data.corr(), annot = True, linewidths = 0.1, cmap = 'Blues')
    plt.title('Numerical Features Correlation')
    plt.show()

