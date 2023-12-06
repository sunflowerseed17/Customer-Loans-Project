import yaml
from sqlalchemy import create_engine
import pandas as pd
import datetime as datetime
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro, zscore
import seaborn as sns

'''
Connecting to the Database and extracting the information to save it to the computer locally 
'''
class RDSDatabaseConnector():


    #Importing credentials
    def import_credentials():
        # Imports credentials from the credentials.yaml file
        with open('credentials.yaml', 'r') as f:
            database = yaml.safe_load(f)
        return database
    
    credentials = import_credentials()

    # Init function 
    def __init__(self):
        credentials = RDSDatabaseConnector.import_credentials()
        self.username = credentials.get('RDS_USER', None)
        self.password = credentials.get('RDS_PASSWORD', None)
        self.host = credentials.get('RDS_HOST', None)
        self.port = credentials.get('RDS_PORT', None)
        self.database_name = credentials.get('RDS_DATABASE', None)
        self.engine = self.create_engine()


    def create_engine(self):
        #Creates the SQLAlchemy engine so that is it easier for us to query the data
        connection_string = f'postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}'
        engine = create_engine(connection_string)
        return engine
    
    # Extraction of database
    
    def extraction_loans(self):
        #Extracting the dataframe as a pandas dataframe
        try:
            sql_query = 'SELECT * FROM loan_payments'
            df = pd.read_sql_query(sql_query, self.engine)
            return df
        except Exception as e:
            print(f"Error extracting loan payments data: {e}")
            self.engine.dispose()
            self.engine = self.create_engine()
            return None
    
    
    def saving_file(self, file_name):
        # Saving the dataframe as a CSV
        df = self.extraction_loans()
        try:
            df.to_csv(file_name)
            print(f"Data saved to {file_name}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")

'''
Creating a class which has static functions to easily transofmr the data types of the columns of the dataframe
'''
class DataTransform():
    
    @staticmethod
    def convert_to_numeric(column):
        # Convert the column to numeric format
        return pd.to_numeric(column, errors='coerce')

    @staticmethod
    def convert_to_datetime(column, date_format='%b-%Y'):
        # Convert the column to datetime format
        return pd.to_datetime(column, format=date_format, errors='coerce')

    @staticmethod
    def convert_to_categorical(column):
        # Convert the column to categorical format
        return column.astype('category')

    @staticmethod
    def remove_symbols(column, symbols_to_remove):
        # Remove specified symbols from the column
        for symbol in symbols_to_remove:
            column = column.str.replace(symbol, '')
        return column

'''
Creating a class to extract information from the DataFrame and its columns 
'''
class DataFrameInfo():

    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe


    # Function which extracts information about the columns of the dataframe
    def column_info(self):
        types = self.my_dataframe.dtypes
        number_of_int_columns = types[types == 'int64'].count()
        number_of_datetime_columns = types[types == 'datetime64[ns]'].count()
        number_of_float_columns = types[types == 'float64'].count()
        number_of_object_columns = types[types == 'object'].count()
        print(f'There are {number_of_int_columns} integer columns, {number_of_datetime_columns} date columns, {number_of_float_columns} float columns and {number_of_object_columns} categorical columns.') 
        print(f'The names and data types of each column are:\n{types}')
        
    # Function which extracts statistical values on the columns
    def columns_basic_stats(self):
        column_stats = self.my_dataframe.describe(include='all')
        return print(column_stats)

    # Function which prints out the shape of the DataFrame
    def data_shape(self):
        shape = self.my_dataframe.shape
        return print(shape)

    # Function which generates the percentages of NULL values in each column
    def null_percentage(self):
        null_percentages_dict = {}
        total_rows = len(self.my_dataframe)
        for column in self.my_dataframe.columns:
            null_counts = self.my_dataframe[column].isna().sum()
            null_per = (null_counts / total_rows) * 100
            null_percentages_dict[column] = null_per
        return print(null_percentages_dict)

'''
Creating a class which will visualise data at different points
'''
class Plotter():

    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe

    #visualising nulls in a helpful way
    def null_plotter(self):
        plt.figure(figsize=(12,8))
        nulls_matrix = msno.matrix(self.my_dataframe, fontsize=10, labels=True )
        return plt.show()
    
    # Function which showcases the distribution shape of the columns
    def normality_check(self):
        to_be_normalised = []
        for column in self.my_dataframe.columns:
            cleaned_data = self.my_dataframe[column].dropna()
            if cleaned_data.dtype in [int, float]: 
                #Showing the data in the form of a histogram
                plt.hist(cleaned_data, edgecolor='black', bins=20)
                plt.title(f'Histogram to see the distribution for {column}'), plt.ylabel('Frequency')
                plt.show()
            # Creating a QQplot to showcase the deviation from normality 
                sm.qqplot(cleaned_data)
                plt.title(f'QQplot to see the distribution for {column}.')
                plt.show()
            #Conducting a Shapiro normality test to see if the data is normally distributed
                normality = shapiro(cleaned_data)
            # Testing the significance of the skewness with a 5% sensitivity
                print(f'The normality value for this is {normality}.')
                if normality.pvalue < 0.05: 
                    print(f'Column name: {column} . This is significant!')
                    to_be_normalised.append(column)
                else: 
                    print(f'Column name: {column} . This is insignificant!')
            else: 
                 print(f"Skipping {column} as it contains non-numeric data.")
            print(f'The columns which need to be normalised are: {to_be_normalised}')

    def visualisee_data(self):
        self.my_dataframe['id'] = self.my_dataframe['id'].astype(str)
        numeric_columns = self.my_dataframe.select_dtypes(include=['int', 'float']).columns
        # visualising the data so that I can assess if there is any outliers visually
        for column in numeric_columns:
            plt.scatter(self.my_dataframe['id'], self.my_dataframe[column])
            plt.title(f'Scatter Plot: ID vs {column}')
            plt.xlabel('ID')
            plt.ylabel(f'{column}')
            plt.show()
    
    def visualise_correlation(self):
        numeric_columns = self.my_dataframe.select_dtypes(include=['float', 'int'])
        column_correlation_matrix = numeric_columns.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(column_correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix Heatmap')
        plt.show()

'''
Creating a class which will perform EDA transformations on the data and impute
'''
class DataFrameTransform():

    def __init__(self,my_dataframe):
        self.my_dataframe = my_dataframe

    def impute_null_values(self, column_name, imputation_method='median'):
    ### Figure out a way to do this based on columns which have >n amount of NA's. This is to avoid having to visualise it before noting down which columns have NaN's
        if imputation_method not in ['median', 'mean', 'mode']:
            raise ValueError("Invalid imputation method. Use 'median' or 'mean' or 'mode'.")

        if column_name not in self.my_dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

        if imputation_method == 'median':
            imputation_value = self.my_dataframe[column_name].median()
        elif imputation_method == 'mean':
            imputation_value = self.my_dataframe[column_name].mean()
        elif imputation_method == 'mode':
            imputation_value = self.my_dataframe[column_name].mode().iloc[0]

        self.my_dataframe[column_name].fillna(imputation_value, inplace=True)

    #checking which columns should be normalised
    def check_normalising_data(self):
        to_be_normalised = []
        for column in self.my_dataframe.columns:
            cleaned_data = self.my_dataframe[column].dropna()
            if cleaned_data.dtype in [int, float]: 
            #Conducting a Shapiro normality test to see if the data is normally distributed
                normality = shapiro(cleaned_data)
            # Testing the significance of the skewness with a 10% sensitivity
                if normality.pvalue < 0.1: 
                    print(f'Column name: {column} . This is significant!')
                    to_be_normalised.append(column)
                else: 
                    print(f'Column name: {column} . This is insignificant!')
            else: 
                 print(f"Skipping {column} as it contains non-numeric data.")
        print(f'The columns which need to be normalised are: {to_be_normalised}')

    #Inputing which columns you want to normalise and the method. (this applies the same method for all inputted)
    def normalising_data(self, columns, normalising_method = 'log'):
        # To reduce right skewness, take roots/logs/reciprocals (roots weakest)
        # To redeuce left skewness, take square roots/ cube roots/ higher powers.
        for column in columns:
            if 'id' in column.lower():
                continue
            if normalising_method not in  ['log', 'square root', 'cube root']:
                return ValueError("Invalid type of normalisation method. Pick between log, square root or cube root")
            elif normalising_method == 'log':
                # Avoiding "RuntimeWarning: divide by zero encountered in log" Error by adding a small value to the data to avoid zero's.
                new_column_name = f"{column}_log"
                self.my_dataframe[new_column_name] = np.log(self.my_dataframe[column] + 1e-10)
            elif normalising_method == 'square root':
                new_column_name = f"{column}_sqrt"
                self.my_dataframe[new_column_name] = np.sqrt(self.my_dataframe[column])
            elif normalising_method == 'cube root':
                new_column_name = f"{column}_cbrt"
                self.my_dataframe[new_column_name] = np.cbrt(self.my_dataframe[column])           
        output_filename = "normalized_data.csv"
        self.my_dataframe.to_csv(output_filename, index=False)
        print(f"Normalized data saved to {output_filename}")

    #Creating a dictionary which lists all the outliers IDs associated with a column, using a z-score analysis

    def get_outliers(self, columns, threshold=3):
        outlier_ids_dict = {}

        for column in columns:
            z_scores = np.abs(zscore(self.my_dataframe[column]))
            outliers_mask = (z_scores > threshold)
            # Get the IDs of outliers
            outlier_ids_dict[column] = self.my_dataframe['id'][outliers_mask].tolist()

        #for column, ids in outlier_ids_dict.items():
            #print(f"Outliers IDs for {column}: {ids}")

        # Finding duplicate IDs which may be problematic
        all_ids = [id for ids in outlier_ids_dict.values() for id in ids]
        df_ids = pd.DataFrame({'id': all_ids})
        duplicate_ids = df_ids[df_ids.duplicated(keep=False)]
        columns_for_duplicate_ids = {}
        for duplicate_id in duplicate_ids['id'].unique():
            columns_for_duplicate_ids[duplicate_id] = []
            for column, ids in outlier_ids_dict.items():
                if duplicate_id in ids:
                    columns_for_duplicate_ids[duplicate_id].append(column)
    
        return columns_for_duplicate_ids

    
    # Actually treating the outliers through inputting a list of IDs which should be removed from the dataset
    def treating_outliers(self, columns_for_duplicate_ids, column_limit=4):
        #  The columns_for_duplicate_ids refers to the dictionary which is returned in the get_outliers() function. the column limit is the amount of times which the id is specified as an outlier in the list of columns. i.e. if the ID is found in 4 or more columns as an outlier, it's removed.
        removed_ids = self.my_dataframe['id'].isin([id for id, columns in columns_for_duplicate_ids.items() if len(columns) >= column_limit])
        filtered_dataframe = self.my_dataframe[~removed_ids]
        filtered_dataframe = filtered_dataframe.reset_index(drop=True)
        print(f"The amount of rows removed is: {removed_ids.sum()}. The total number of rows/IDs now is {len(filtered_dataframe.axes[0])}")
        return filtered_dataframe

    # Creating a correlation matrix and 
    def correlation_matrix_analysis(self):
        numeric_columns = self.my_dataframe.select_dtypes(include=['float', 'int'])
        column_correlation_matrix = numeric_columns.corr()
        correlation_threshold = 0.8
        highly_correlated_pairs = (column_correlation_matrix.abs() > correlation_threshold) & (column_correlation_matrix != 1)
        highly_correlated_columns = set()
        for col in highly_correlated_pairs.columns:
            correlated_cols = highly_correlated_pairs.index[highly_correlated_pairs[col]].tolist()
            highly_correlated_columns.update({col, *correlated_cols})
        print("Correlation Matrix for All Columns:")
        print(column_correlation_matrix)
        print("\nHighly Correlated Columns:")
        print(highly_correlated_columns)    
