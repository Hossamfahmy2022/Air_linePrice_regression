import numpy as np
import pandas as pd
# preprocessing and feature transformation 
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.impute import SimpleImputer
from category_encoders.binary import BinaryEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


df=pd.read_excel("Data_Train.xlsx")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True,inplace=True)
valid_airline=df["Airline"].value_counts()[df["Airline"].value_counts()>13].index.to_list()
def handle_airline(value):
    if value in valid_airline:
        return value
    else:
        return "other"    
df["Airline"]=df["Airline"].apply(handle_airline)
df["Date_of_Journey"]=pd.to_datetime(df["Date_of_Journey"])
df["Day_of_Journey"]=df["Date_of_Journey"].dt.day_name()
df["Month_of_Journey"]=df["Date_of_Journey"].dt.month_name()
df["Year_of_Journey"]=df["Date_of_Journey"].dt.year
df.drop(['Route'],axis=1,inplace=True)
df['Dep_Time']=pd.to_datetime(df['Dep_Time'])
df['Dep_Time']=df['Dep_Time'].dt.hour
def handle_Dep_Time(time):
    try:
        if (time >=1) & (time <=12):
            return "Am"
        else:
            return "Pm"    
    except:
        return np.nan     
df['Dep_Time_Hour']=df['Dep_Time'].apply(handle_Dep_Time)
df.drop("Dep_Time",axis=1,inplace=True)
df['Arrival_Time']=pd.to_datetime(df['Arrival_Time'])
df[(df['Arrival_Time'].dt.month < df["Date_of_Journey"].dt.month) | (df['Arrival_Time'].dt.day < df["Date_of_Journey"].dt.day)][["Date_of_Journey",'Arrival_Time']]
error_index=df[(df['Arrival_Time'].dt.month < df["Date_of_Journey"].dt.month) | (df['Arrival_Time'].dt.day < df["Date_of_Journey"].dt.day)].index.to_list()
df.drop(index=error_index,axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)
df.drop(columns=["Date_of_Journey"],axis=1,inplace=True)
df["Arrival_day"]=df["Arrival_Time"].dt.day_name()
df["Arrival_month"]=df["Arrival_Time"].dt.month_name()
df["Arrival_Hour"]=df["Arrival_Time"].dt.hour
def handle_Arrival_Hour(time):
    try:
        if (time >=1) & (time <=12):
            return "Am"
        else:
            return "Pm"    
    except:
        return np.nan     
df['Arrival_Hour']=df['Arrival_Hour'].apply(handle_Arrival_Hour)
df.drop(columns=["Arrival_Time"],axis=1,inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True,inplace=True)
df["Duration"]=df["Duration"].str.split(" ")
def handle_Duration(value):
    try:
        if len(value)>1:
            hours=int(value[0].split("h")[0])
            minutes=int(value[1].split("m")[0])
            return (hours*60)+minutes
        else:
            hours=int(value[0].split("h")[0])
            return (hours*60)
    except:
        return np.NaN
df["Duration_with_minutes"]=df["Duration"].apply(handle_Duration)     
df.drop("Duration",axis=1,inplace=True)
def handle_Total_Stops(value):
    try:
        if value=="non-stop":
            return 0
        else:
            return value[:2]    
    except:
        return np.NaN
df["Total_Stops"]=df["Total_Stops"].apply(handle_Total_Stops)            
df["Total_Stops"]=df["Total_Stops"].astype("int")
df.drop("Additional_Info",axis=1,inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True,inplace=True)
df.drop(columns=['Year_of_Journey','Arrival_month','Arrival_Hour','Arrival_day'],axis=1,inplace=True)
# split data 
X=df.drop("Price",axis=1)
y=df["Price"]
cat_columns=X.select_dtypes(include="object").columns.to_list()
num_columns=X.select_dtypes(exclude="object").columns.to_list()
y=np.log1p(df["Price"])
num_pipe=Pipeline(steps=[
    ("impute",SimpleImputer(strategy="mean")),
    ("tranforme",PowerTransformer(method="yeo-johnson",standardize=True))
])
cat_pipe=Pipeline(steps=[
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("encode",BinaryEncoder())
])
# Define the column transformer with multiple operations for numeric features
preprocessor = ColumnTransformer([
    ('numeric', num_pipe, num_columns),
    ('categorical', cat_pipe, cat_columns)
])

# Define the full pipeline including the preprocessor and the model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor)  
])


_ =full_pipeline.fit(X)




def process_new(X_new):
    ''' This Function is to apply the pipeline to user data. Taking a list.
    
    Args:
    *****
        (X_new: List) --> The users input as a list.

    Returns:
    *******
        (X_processed: 2D numpy array) --> The processed numpy array of userf input.
    '''
    
    ## To DataFrame
    df_new = pd.DataFrame([X_new])
    df_new.columns = X.columns

    ## Adjust the Datatypes
    df_new['Airline'] = df_new['Airline'].astype('str')
    df_new['Source'] = df_new['Source'].astype('str')
    df_new['Destination'] = df_new['Destination'].astype('str')
    df_new['Total_Stops'] = df_new['Total_Stops'].astype('int')
    df_new['Day_of_Journey'] = df_new['Day_of_Journey'].astype('str')
    df_new['Month_of_Journey'] = df_new['Month_of_Journey'].astype('str')
    df_new['Dep_Time_Hour'] = df_new['Dep_Time_Hour'].astype('str')
    df_new['Duration_with_minutes'] = df_new['Duration_with_minutes'].astype('int')
 
    ## Apply the pipeline
    X_processed = full_pipeline.transform(df_new)


    return X_processed






