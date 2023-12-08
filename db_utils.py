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
from dateutil.relativedelta import relativedelta
import os

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

'''
Creating a class which runs the queries
'''
class Query():
    
    def __init__(self,my_dataframe):
        self.my_dataframe = my_dataframe

    #Recoveries as a percentage of the amounts funded (total and by investors) 
    def percentage_recovered(self):
        total_recovered = self.my_dataframe['total_payment'].sum()
        total_recovered_inv = self.my_dataframe['total_payment_inv'].sum()
        total_funded = self.my_dataframe['funded_amount'].sum()
        total_funded_inv = self.my_dataframe['funded_amount_inv'].sum()
        perc_rec = total_recovered / total_funded * 100
        perc_rec_inv = total_recovered_inv / total_funded_inv * 100
        return perc_rec, perc_rec_inv
    
    def percentages_loss(self):
        numerator_total = self.my_dataframe[self.my_dataframe['loan_status'] == "Charged Off"].shape[0]
        denominator_total = self.my_dataframe.shape[0]
        percentage = (numerator_total / denominator_total) * 100
        return percentage
    
    def percentage_loss_year(self):
        self.my_dataframe['issue_date'] = pd.to_datetime(self.my_dataframe['issue_date'], format='%d/%m/%Y')
        self.my_dataframe['year'] = self.my_dataframe['issue_date'].dt.year
        unique_years = self.my_dataframe['year'].unique()
        for year in unique_years:
            year_df = self.my_dataframe[self.my_dataframe['year'] == year].copy()
            charged_off_count = (year_df['loan_status'] == 'Charged Off').sum()
            total_loans_count = year_df.shape[0]
            percentage_losses = (charged_off_count / total_loans_count) * 100
            print(f"For loans issued in {year}, the percentage of losses is: {percentage_losses:.2f}%")
    
    def amount_paid_before_loss(self):
        charged_off = self.my_dataframe[self.my_dataframe['loan_status'] == "Charged Off"].copy()
        amount_paid = round(charged_off['total_payment'].sum(),2)
        amount_borrowed = round(charged_off['loan_amount'].sum(),2)
        return print(f'The total amount paid back was {amount_paid} out of {amount_borrowed} total borrowed for these accounts.')
    
    def projected_loss(self):
        charged_off = self.my_dataframe[self.my_dataframe['loan_status'] == "Charged Off"].copy()
        amount_paid = round(charged_off['total_payment'].sum(),2)
        potential_loss_nxtyr = (charged_off['funded_amount'] * (1 + charged_off['int_rate'] / 100)).sum()
        recovery_amount = charged_off['recoveries'].sum() + charged_off['collection_recovery_fee'].sum()
        projected_loss = potential_loss_nxtyr - amount_paid - recovery_amount
        return print(f"The projected loss on the loans marked as CHARGED OFF for next year is £{round(projected_loss, 2)}")

    def revenue_loss(self):
        charged_off = self.my_dataframe[self.my_dataframe['loan_status'] == "Charged Off"].copy()
        amount_paid = round(charged_off['total_payment'].sum(),2)
        three_year_loans_co = self.my_dataframe[(self.my_dataframe['term'] == "36 months") & (self.my_dataframe['loan_status'] == 'Charged Off')].copy()
        five_year_loans_co = self.my_dataframe[(self.my_dataframe['term'] == "60 months") & (self.my_dataframe['loan_status'] == 'Charged Off')].copy()
        three_year_loans_funded = (three_year_loans_co['funded_amount'] * ((1 + three_year_loans_co['int_rate']/100) ** 3)).sum()
        five_year_loans_funded = (five_year_loans_co['funded_amount'] * ((1 + five_year_loans_co['int_rate']/100) ** 5)).sum()
        recovery_amount = charged_off['recoveries'].sum() + charged_off['collection_recovery_fee'].sum()
        projected_revenue = five_year_loans_funded + three_year_loans_funded 
        revenue_lost = projected_revenue - amount_paid - recovery_amount
        return print(f" The loss of revenue the loans marked as CHARGED OFF would have generated would have been £{round(revenue_lost,2)}")

    def cumulative_revenue_loss(self):
        copy = self.my_dataframe.copy()
        charged_off = copy[copy['loan_status'] == "Charged Off"].copy()
        charged_off['issue_date'] = pd.to_datetime(charged_off['issue_date'], format='%d/%m/%Y')
        charged_off['last_payment_date'] = pd.to_datetime(charged_off['last_payment_date'], format='%d/%m/%Y')
        charged_off['term_completed'] = (charged_off['last_payment_date'] - charged_off['issue_date']).dt.days // 30
        charged_off['term_left'] = np.where(charged_off['term'] == '36 months', 36 - charged_off['term_completed'], 60 - charged_off['term_completed'])
        cumulative_revenue_lost = 0
        monthly_revenue_lost = []
        for i in range(1, (charged_off['term_left'].max()+1)): 
            charged_off = charged_off[charged_off['term_left']>0].copy()
            cumulative_revenue_lost += charged_off['instalment'].sum() 
            monthly_revenue_lost.append(cumulative_revenue_lost) 
            charged_off['term_left'] -= 1
        return monthly_revenue_lost
    
    def percentage_late(self):
        late_df = (self.my_dataframe['loan_status'].str.contains('Late', case=False, na=False))
        late = late_df.sum()
        all = self.my_dataframe.shape[0]
        late_percentage = late / all * 100
        return print(f"The percentage of current risk users from all the loans is {round(late_percentage,2)}%")
    
    def total_risk_and_potential_loss(self):
        late_df = self.my_dataframe[self.my_dataframe['loan_status'].str.contains('Late', case=False, na=False)].copy()
        late = late_df.shape[0]
        print(f"The amount of people in the Risk bracket is {late}")
        amount_paid = round(late_df['total_payment'].sum(),2)
        total_funded_amount = late_df['funded_amount'].sum()
        immediate_loss = total_funded_amount - amount_paid
        print(f"The immediate loss if the Risk customers were charged off would be £{round(immediate_loss,2)}")
        three_year_loans_late = late_df[late_df['term'] == "36 months"].copy()
        five_year_loans_late = late_df[late_df['term'] == "60 months"].copy()
        three_year_loans_funded = (three_year_loans_late['funded_amount'] * ((1 + three_year_loans_late['int_rate']/100) ** 3)).sum()
        five_year_loans_funded = (five_year_loans_late['funded_amount'] * ((1 + five_year_loans_late['int_rate']/100) ** 5)).sum()
        projected_revenue = five_year_loans_funded + three_year_loans_funded
        potential_lost = projected_revenue - amount_paid
        return print(f" The potential loss if the customers in this bracket were charged off is £{round(potential_lost,2)}")
    
    def percentage_total_revenue_lost(self):
        copy_df = self.my_dataframe.copy()
        loss = copy_df[(copy_df['loan_status'] == "Charged Off")|(copy_df['loan_status'].str.contains('Late', case=False, na=False))].copy()
        loss['issue_date'] = pd.to_datetime(loss['issue_date'], format='%d/%m/%Y')
        loss['last_payment_date'] = pd.to_datetime(loss['last_payment_date'], format='%d/%m/%Y')
        loss['term_completed'] = (loss['last_payment_date'] - loss['issue_date']).dt.days // 30
        loss['term_left'] = np.where(loss['term'] == '36 months', 36 - loss['term_completed'], 60 - loss['term_completed'])
        cumulative_revenue_lost = 0
        monthly_revenue_lost = []
        for i in range(1, (loss['term_left'].max()+1)): 
            loss = loss[loss['term_left']>0] 
            cumulative_revenue_lost += loss['instalment'].sum() 
            monthly_revenue_lost.append(cumulative_revenue_lost) 
            loss['term_left'] -= 1
        copy_df['issue_date'] = pd.to_datetime(copy_df['issue_date'], format='%d/%m/%Y')
        copy_df['last_payment_date'] = pd.to_datetime(copy_df['last_payment_date'], format='%d/%m/%Y')
        copy_df['term_completed'] = (copy_df['last_payment_date'] - copy_df['issue_date']).dt.days // 30
        copy_df['term_left'] = np.where(copy_df['term'] == '36 months', 36 - copy_df['term_completed'], 60 - copy_df['term_completed'])
        cumulative_potential_revenue = 0
        potential_monthly_revenue = []
        for i in range(1, (copy_df['term_left'].max())):
            copy_df = copy_df[copy_df['term_left']>0] 
            cumulative_potential_revenue += copy_df['instalment'].sum() 
            potential_monthly_revenue.append(cumulative_potential_revenue) 
            copy_df['term_left'] -= 1
        percentages_revenue_lost = []
        for i in range(len(potential_monthly_revenue)):
            if potential_monthly_revenue[i] != 0:
                percentage = monthly_revenue_lost[i] / potential_monthly_revenue[i] * 100
                percentages_revenue_lost.append(percentage)
            else:
                percentages_revenue_lost.append(0)
        return percentages_revenue_lost
    
    def indicator_of_loss_analysis(self,file_name):
        charged_off_dataframe = self.my_dataframe[self.my_dataframe['loan_status'] == "Charged Off"].copy()
        full_file_path = os.path.join(os.getcwd(), file_name)

        if not os.path.exists(full_file_path):
            charged_off_dataframe.to_csv(full_file_path, index=False)
            print(f"DataFrame saved as {full_file_path}")
        else:
            print(f"A file with the name {full_file_path} already exists. DataFrame not saved.")

    def indicator_of_potential_loss(self, file_name):
        late_df = self.my_dataframe[self.my_dataframe['loan_status'].str.contains('Late', case=False, na=False)].copy()
        full_file_path = os.path.join(os.getcwd(), file_name)

        if not os.path.exists(full_file_path):
            late_df.to_csv(full_file_path, index=False)
            print(f"DataFrame saved as {full_file_path}")
        else:
            print(f"A file with the name {full_file_path} already exists. DataFrame not saved.")



'''
Creating a class which visualises the queries 
'''
class QueryVisualiser():

    def __init__(self, my_dataframe):
        self.my_dataframe = my_dataframe

    def percentage_visuals(self, perc_rec, perc_rec_inv):
        labels = ['Investor Recovered', 'Total Recovered']
        percentages = [perc_rec_inv, perc_rec]
        fig, ax = plt.subplots(figsize = (10,6))
        bars = ax.bar(labels, percentages, color=['blue', 'blue'])
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=5)
        plt.ylabel('Percentage')
        plt.title('Recovery Percentages for Investors and Total')
        plt.show()

    def cumulative_revenue_loss(self, monthly_revenue_lost):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(monthly_revenue_lost) + 1), monthly_revenue_lost, marker='o', linestyle='-', color='b')
        plt.title('Cumulative Revenue Lost Over Time')
        plt.xlabel('Months')
        plt.ylabel('Cumulative Revenue Lost (USD)')
        plt.grid(True)
        plt.show()

    def cumulative_potential_revenue_loss(self, percentages_revenue_lost):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(percentages_revenue_lost) + 1), percentages_revenue_lost, marker='o', linestyle='-', color='b')
        plt.title('Percentage of total expected revenue which is accounted for by customers who are at risk or have already defaulted')
        plt.xlabel('Months')
        plt.ylabel('Percentage per month')
        plt.grid(True)
        plt.show()
