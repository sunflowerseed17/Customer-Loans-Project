# Customer-Loans-Project

The project showcases the methods of cleaning and analysing large datasets, specifically for loan-related datasets. 

## Installation

1) Clone the repository to your local device
2) Running data_file.csv through data_file_transformation.ipynb cleans the data and creates the renditions of the database at different stages
3) Running the appropriate .csv file through the querying_data.ipynb outputs important queries about the data

## This project takes a database in the .csv format (specifically a database of financial and loan information).
 The original database is stored at data_file.csv.

_The data_file_transformation.ipynb incrementally cleans the data at each stage through:_
- Converting columns into the correct data formats
- Assessing the missing data and what columns may need to be dropped in relation to this.
- Imputing missing values which are applicable to imputation
- Checking the normality of the columns and conducting transformations in order to normalise them
- Visualising the data
- Treating the data through removal of outliers
- Checking for correlations between columns and removing columns as applicable

There are checkpoints at which the dataframe is saved after its transformations. 
In this case normalized_data.csv is created after normality transformation have been applied.
filtered_dataframe.csv is created at the end of the correlation analysis. 

## Data analysis

The querying_data.ipynb is the notebook to use in order to query the data. 
It works side-by-side with the Query class and VisualiseQuery class in db_utils.py in order to provide functions for analysing the data. 


## List of definitions

Here is the list of definitions for each data-file.csv columns:
<details open>
    <summary>List of definitions for each column </summary>
    <ul>
        <li> **id**: unique id of the loan </li>
        <li> **member_id**: id of the member to took out the loan</li>
        <li> **loan_amount**: amount of loan the applicant received</li>
        <li> **funded_amount**: The total amount committed to the loan at the point in time </li>
        <li> **funded_amount_inv**: The total amount committed by investors for that loan at that point in time</li> 
        <li> **term**: The number of monthly payments for the loan</li>
        <li>**int_rate**: Interest rate on the loan</li>
        <li> **instalment**: The monthly payment owned by the borrower</li>
        <li> **grade**: LC assigned loan grade</li>
        <li> **sub_grade**: LC assigned loan sub grade</li>
        <li> **employment_length**: Employment length in years.</li>
        <li> **home_ownership**: The home ownership status provided by the borrower</li>
        <li> **annual_inc**: The annual income of the borrower</li>
        <li> **verification_status**: Indicates whether the borrowers income was verified by the LC or the income source was verified</li>
        <li> **issue_date:** Issue date of the loan</li>
        <li> **loan_status**: Current status of the loan</li>
        <li> **payment_plan**: Indicates if a payment plan is in place for the loan. Indication borrower is struggling to pay.</li>
        <li> **purpose**: A category provided by the borrower for the loan request.</li>
        <li> **dti**: A ratio calculated using the borrowerâ€™s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowerâ€™s self-reported monthly income.</li>
        <li> **delinq_2yr**: The number of 30+ days past-due payment in the borrower's credit file for the past 2 years.</li>
        <li> **earliest_credit_line**: The month the borrower's earliest reported credit line was opened</li>
        <li> **inq_last_6mths**: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)</li>
        <li> **mths_since_last_record**: The number of months since the last public record.</li>
        <li> **open_accounts**: The number of open credit lines in the borrower's credit file.</li>
        <li> **total_accounts**: The total number of credit lines currently in the borrower's credit file</li>
        <li> **out_prncp**: Remaining outstanding principal for total amount funded</li>
        <li> **out_prncp_inv**: Remaining outstanding principal for portion of total amount funded by investors</li>
        <li> **total_payment**: Payments received to date for total amount funded</li>
        <li> **total_rec_int**: Interest received to date</li>
        <li> **total_rec_late_fee**: Late fees received to date</li>
        <li> **recoveries**: post charge off gross recovery</li>
        <li> **collection_recovery_fee**: post charge off collection fee</li>
        <li> **last_payment_date**: Last month payment was received</li>
        <li> **last_payment_amount**: Last total payment amount received</li>
        <li> **next_payment_date**: Next scheduled payment date</li>
        <li> **last_credit_pull_date**: The most recent month LC pulled credit for this loan</li>
        <li> **collections_12_mths_ex_med**: Number of collections in 12 months excluding medical collections</li>
        <li> **mths_since_last_major_derog**: Months since most recent 90-day or worse rating</li>
        <li> **policy_code**: publicly available policy_code=1 new products not publicly available policy_code=2</li>
        <li> **application_type**: Indicates whether the loan is an individual application or a joint application with two co-borrowers</li>
    </ul>
</details>
