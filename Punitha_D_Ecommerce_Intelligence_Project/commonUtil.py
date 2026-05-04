import pandas as pd

#### Load each dataset
####Inspect structure using .head(), .info(), .describe() step-1
def readDataset(datasetName):
    fileData = pd.read_csv(datasetName)
    infoDetails = fileData.info()
    headDetails = fileData.head()
    describeDetails = fileData.describe()
    # print(f"### Load {datasetName} data ### \n", fileData)
    # print(f"### Info details of {datasetName} data ### \n", infoDetails)
    # print(f"### head details of {datasetName} data ### \n", headDetails)
    # print(f"### describe details of {datasetName} data ### \n", describeDetails)
    return fileData
    
### Identify primary

def identify_primary_key_column(tableName):
    for col in tableName.columns:
        if(tableName[col].is_unique and tableName[col].notnull().all()):
            colName = col
            msg = "is a primary key"
            return colName, msg
            
# #Handle missing values appropriately 
def handle_missing_values(tblName, name):
    print("Before handling", tblName.isnull().sum())
    if(tblName.isnull().values.any()):
        tblName = tblName.bfill()
        message = "After filling for {name} dataset"
    else:
        tblName= tblName.isnull().sum()
        message = "After handling Nochange"
    return tblName, message
# #Remove duplicated values
def remove_duplidateddsdsd(tblName, name):
    #print(f"Duplicates before removing for {name} dataset", tblName.duplicated().shape(0))
    print(f"total data for {name} dataset", tblName.shape[0])

    if(tblName.duplicated().shape[0] > 0):
       newTable = tblName.drop_duplicates()
       message = "After removing duplicates of {name} dataset"
    else:
        newTable = tblName
        message = "No data to remove duplicates"
    return newTable, message
def remove_duplidated(tblName, name):

    print(f"total data for {name} dataset: {tblName.shape[0]}")

    duplicate_count = tblName.duplicated().sum()

    print(f"Duplicates before removing for {name} dataset: {duplicate_count}")

    if duplicate_count > 0:
        newTable = tblName.drop_duplicates()
    else:
        newTable = tblName

    return newTable



    
#Validate data types and ranges
def verify_date_column(tableName, name):
    for col in tableName.columns:
        if pd.api.types.is_string_dtype(tableName[col]):
            colName1 = col
            colmsg = "is already a date timeformat"
            convertDatetime = ""
            return colName1, colmsg, convertDatetime
        else:
            convertDatetime = pd.to_datetime(tableName[col], errors='coerce')
            colName1 = col
            colmsg = "is already a date timeformat"  
            return colName1, colmsg, convertDatetime
            
#Validate data types and ranges
def validate_customers_column (tableName):
    if tableName["customer_id"].is_unique:
        print("All customer_id are unique")
    if (tableName["customer_zip_code_prefix"].astype(str).str.len() != 5).any():
        print("Customer_zip_code_prefix states are not 2 letters")
    if not (tableName["customer_state"].str.len() == 2).any():
        print("customer_state states are not 2 letters")
    else:
        print("All customer_state states are in correct format")

##Standardize column names if required customers dataset
def standarize_columns(tableName):
    tableName.columns = (
        tableName.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return tableName