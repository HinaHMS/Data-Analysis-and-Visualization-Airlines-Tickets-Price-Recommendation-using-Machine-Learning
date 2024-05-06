#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


train_data = pd.read_excel(r"/Users/csuftitan/Desktop/Flight_Prediction/Data_Train.xlsx")


# In[5]:


train_data.head(4)


# In[6]:


train_data.tail(4)


# In[7]:


train_data.info()


# In[8]:


train_data.isnull()


# In[9]:


train_data.isnull().sum()


# In[10]:


train_data['Total_Stops'].isnull()


# In[11]:


train_data[train_data['Total_Stops'].isnull()]


# In[12]:


train_data.dropna(inplace=True)


# In[13]:


train_data[train_data['Total_Stops'].isnull()]


# In[14]:


train_data.isnull().sum()


# In[15]:


train_data.dtypes


# In[16]:


train_data.info()


# In[ ]:


train_data.info(memory_usage = 'deep')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


data = train_data.copy()


# In[18]:


data.columns


# In[19]:


data.head(3)


# In[20]:


data.dtypes


# In[21]:


def change_into_Datetime(col):
    data[col] = pd.to_datetime(data[col])


# In[22]:


import warnings
from warnings import filterwarnings
filterwarnings("ignore")


# In[23]:


data.columns


# In[24]:


for feature in ['Dep_Time', 'Arrival_Time' , 'Date_of_Journey']:
    change_into_Datetime(feature)


# In[25]:


data.dtypes


# In[26]:


data["Journey_day"] = data['Date_of_Journey'].dt.day


# In[27]:


data["Journey_month"] = data['Date_of_Journey'].dt.month


# In[28]:


data["Journey_year"] = data['Date_of_Journey'].dt.year


# In[29]:


data.head(3)


# In[30]:


def extract_hour_min(df , col):
    df[col+"_hour"] = df[col].dt.hour
    df[col+"_minute"] = df[col].dt.minute
    return df.head(3)


# In[31]:


data.columns


# In[32]:


extract_hour_min(data , "Dep_Time")


# In[33]:


extract_hour_min(data , "Arrival_Time")


# In[34]:


cols_to_drop = ['Arrival_Time' , "Dep_Time"]

data.drop(cols_to_drop , axis=1 , inplace=True )


# In[35]:


data.head(3)


# In[36]:


data.shape


# In[37]:


#### Converting the flight Dep_Time into proper time i.e. mid_night, morning, afternoon and evening.

def flight_dep_time(x):
    if (x>4) and (x<=8):
        return "Early Morning"

    elif (x>8) and (x<=12):
        return "Morning"

    elif (x>12) and (x<=16):
        return "Noon"

    elif (x>16) and (x<=20):
        return "Evening"

    elif (x>20) and (x<=24):
        return "Night"

    else:
        return "late night"


# In[38]:


#### data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind="bar" , color="g")


# In[39]:


data['Dep_Time_hour'].apply(flight_dep_time)


# In[40]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts()


# In[41]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind="bar" , color="g")


# In[42]:


## how to use Plotly interactive plots directly with Pandas dataframes, First u need below set-up !

import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot , iplot , init_notebook_mode , download_plotlyjs
init_notebook_mode(connected=True)
cf.go_offline()

## plot is a command of Matplotlib which is more old-school. It creates static charts
## iplot is an interactive plot. Plotly takes Python code and makes beautiful looking JavaScript plots.


# In[43]:


get_ipython().system('pip install cufflinks')


# In[44]:


## how to use Plotly interactive plots directly with Pandas dataframes, First u need below set-up !

import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot , iplot , init_notebook_mode , download_plotlyjs
init_notebook_mode(connected=True)
cf.go_offline()

## plot is a command of Matplotlib which is more old-school. It creates static charts
## iplot is an interactive plot. Plotly takes Python code and makes beautiful looking JavaScript plots.


# In[45]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind="bar")


# In[46]:


#### More cleaning for Duration of the flight


# In[47]:


def preprocess_duration(x):
    if 'h' not in x:
        x = '0h' + ' ' + x
    elif 'm' not in x:
        x = x + ' ' +'0m'

    return x


# In[48]:


data['Duration'] = data['Duration'].apply(preprocess_duration)


# In[49]:


data.head(3)


# In[50]:


data['Duration']


# In[51]:


data['Duration'][0]


# In[52]:


data['Duration'][0].split(' ')


# In[53]:


data['Duration'][0].split(' ')[0]


# In[54]:


data['Duration'][0].split(' ')[0].split('h')


# In[55]:


'2h 50m'.split(' ')


# In[56]:


'2h 50m'.split(' ')[0]


# In[57]:


'2h 50m'.split(' ')[0][0:-1]


# In[58]:


'2h 50m'.split(' ')[1][0:-1]


# In[59]:


data['Duration_hours'] = data['Duration'].apply(lambda x: int(x.split(' ')[0][0:-1]))


# In[60]:


data['Duration_mins'] = data['Duration'].apply(lambda x: int(x.split(' ')[1][0:-1]))


# In[61]:


data['Duration_hours']


# In[62]:


data.head(4)


# In[63]:


### Lets Analyse whether Duration impacts Price or not ?


# In[64]:


data['Duration'] ## convert duration into total minutes duration ..


# In[65]:


eval('2*60')


# In[66]:


data['Duration_total_mins'] = data['Duration'].str.replace('h' ,"*60").str.replace(' ' , '+').str.replace('m' , "*1").apply(eval)


# In[67]:


data['Duration_total_mins']


# In[68]:


data.columns


# In[69]:


sns.scatterplot(x = 'Duration_total_mins', y = 'Price', data = data)


# In[70]:


sns.lmplot(x = 'Duration_total_mins', y = 'Price', data = data)


# In[71]:


sns.scatterplot(x = 'Duration_total_mins', y = 'Price', hue = 'Total_Stops' , data = data)


# In[72]:


###  on which route Jet Airways is extremely used?


# In[73]:


data['Airline'] == 'IndiGo'


# In[74]:


data[data['Airline'] == 'IndiGo']


# In[75]:


data[data['Airline'] == 'IndiGo'].groupby('Route').size().sort_values(ascending=False)


# In[76]:


### Performing Airline vs Price Analysis..


# In[77]:


data.columns


# In[78]:


sns.boxplot(y='Price' , x='Airline' , data=data.sort_values('Price' , ascending=False))


# In[79]:


sns.boxplot(y='Price' , x='Airline' , data=data.sort_values('Price' , ascending=False))
plt.xticks(rotation="vertical")


# In[80]:


sns.boxplot(y='Price' , x='Airline' , data=data.sort_values('Price' , ascending=False))
plt.xticks(rotation="vertical")
plt.show()


# In[81]:


data.head(2)


# In[82]:


cat_col = [col for col in data.columns if data[col].dtype=="object"]


# In[83]:


num_col = [col for col in data.columns if data[col].dtype!="object"]


# In[84]:


cat_col


# In[85]:


data['Source'].unique()


# In[86]:


data['Source'].apply(lambda x : 1 if x=='Banglore' else 0)


# In[87]:


for sub_category in data['Source'].unique():
    data['Source_'+sub_category] = data['Source'].apply(lambda x : 1 if x==sub_category else 0)


# In[88]:


data.head(3)


# In[89]:


cat_col


# In[90]:


data['Airline'].nunique()


# In[91]:


data.groupby(['Airline'])['Price'].mean().sort_values()


# In[92]:


airlines = data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[93]:


airlines


# In[94]:


dict_airlines = {key:index for index , key in enumerate(airlines , 0)}


# In[95]:


dict_airlines


# In[96]:


data['Airline'] = data['Airline'].map(dict_airlines)


# In[97]:


data['Airline']


# In[98]:


data.head(3)


# In[99]:


data['Destination'].unique()


# In[100]:


data['Destination'].replace('New Delhi' , 'Delhi' , inplace=True)


# In[101]:


data['Destination'].unique()


# In[102]:


dest = data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[103]:


dest


# In[104]:


dict_dest = {key:index for index , key in enumerate(dest , 0)}


# In[105]:


dict_dest


# In[106]:


data['Destination'] = data['Destination'].map(dict_dest)


# In[107]:


data['Destination']


# In[108]:


data.head(3)


# In[109]:


#### Perform Label(Manual) Encoding on Data


# In[110]:


data.head(3)


# In[111]:


data['Total_Stops']


# In[112]:


data['Total_Stops'].unique()


# In[113]:


stop = {'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[114]:


data['Total_Stops'] = data['Total_Stops'].map(stop)


# In[115]:


data['Total_Stops']


# In[116]:


data.head(1)


# In[117]:


data.columns


# In[118]:


data['Additional_Info'].value_counts()/len(data)*100


# In[119]:


data.drop(columns = ['Route','Duration','Date_of_Journey','Additional_Info','Duration_total_mins','Source'])


# In[120]:


data['Journey_year'].unique()


# In[121]:


data.drop(columns=['Date_of_Journey' , 'Additional_Info' , 'Duration_total_mins' , 'Source' , 'Journey_year'] , axis=1 , inplace=True)


# In[122]:


data.head(4)


# In[123]:


data.drop(columns=['Route','Duration'] , axis=1 , inplace=True)


# In[124]:


data.head(4)


# In[125]:


##.. Perform outlier detection !


# In[126]:


def plot(df, col):
    fig , (ax1 , ax2 , ax3) = plt.subplots(3,1)

    sns.distplot(df[col] , ax=ax1)
    sns.boxplot(df[col] , ax=ax2)
    sns.distplot(df[col] , ax=ax3 , kde=False)


# In[127]:


plot(data , 'Price')


# In[128]:


q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)

iqr = q3- q1

maximum = q3 + 1.5*iqr
minimum = q1 - 1.5*iqr


# In[129]:


print(maximum)


# In[130]:


print(minimum)


# In[131]:


print([price for price in data['Price'] if price> maximum or price<minimum])


# In[132]:


len([price for price in data['Price'] if price> maximum or price<minimum])


# In[133]:


### wherever I have price >35K just replace replace it with median of Price

data['Price'] = np.where(data['Price']>=35000 , data['Price'].median() , data['Price'])


# In[134]:


print([price for price in data['Price'] if price> maximum or price<minimum])


# In[135]:


plot(data , 'Price')


# In[ ]:





# In[136]:


#### Perform feature selection


# In[137]:


X = data.drop(['Price'] , axis=1)


# In[138]:


y = data['Price']


# In[139]:


from sklearn.feature_selection import mutual_info_regression


# In[140]:


imp = mutual_info_regression(X , y)


# In[141]:


imp


# In[142]:


imp_df = pd.DataFrame(imp , index=X.columns)


# In[143]:


imp_df.columns = ['importance']


# In[144]:


imp_df


# In[145]:


imp_df.sort_values(by='importance' , ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:


##.. Lets Build ML model


# In[146]:


from sklearn.model_selection import train_test_split


# In[147]:


X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.25, random_state=42)


# In[148]:


from sklearn.ensemble import RandomForestRegressor


# In[149]:


ml_model = RandomForestRegressor()


# In[150]:


ml_model.fit(X_train , y_train)


# In[151]:


y_pred = ml_model.predict(X_test)


# In[152]:


y_pred


# In[153]:


from sklearn import metrics


# In[154]:


metrics.r2_score(y_test , y_pred)


# In[155]:


#### Save model


# In[157]:


get_ipython().system('pip install pickle')


# In[158]:


import pickle


# In[159]:


# open a file, where you want to store the data
file = open(r'/Users/csuftitan/Desktop/Flight_Prediction/rf_random.pkl' , 'wb')


# In[160]:


# dump information to that file
pickle.dump(ml_model , file)


# In[161]:


model = open(r'/Users/csuftitan/Desktop/Flight_Prediction/rf_random.pkl' , 'rb')


# In[162]:


forest = pickle.load(model)


# In[163]:


y_pred2 = forest.predict(X_test)


# In[164]:


metrics.r2_score(y_test , y_pred2)


# In[ ]:





# In[165]:


##### Automate ML Pipeline


# In[166]:


##### Define Evaluation Metric


# In[167]:


def mape(y_true , y_pred):
    y_true , y_pred = np.array(y_true) , np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[168]:


mape(y_test , y_pred)


# In[169]:


from sklearn import metrics


# In[170]:


def predict(ml_model):
    model = ml_model.fit(X_train , y_train)
    print('Training score : {}'.format(model.score(X_train , y_train)))
    y_predection = model.predict(X_test)
    print('predictions are : {}'.format(y_predection))
    print('\n')
    r2_score = metrics.r2_score(y_test , y_predection)
    print('r2 score : {}'.format(r2_score))
    print('MAE : {}'.format(metrics.mean_absolute_error(y_test , y_predection)))
    print('MSE : {}'.format(metrics.mean_squared_error(y_test , y_predection)))
    print('RMSE : {}'.format(np.sqrt(metrics.mean_squared_error(y_test , y_predection))))
    print('MAPE : {}'.format(mape(y_test , y_predection)))
    sns.distplot(y_test - y_predection)


# In[171]:


predict(RandomForestRegressor())


# In[172]:


from sklearn.tree import DecisionTreeRegressor


# In[173]:


predict(DecisionTreeRegressor())


# In[174]:


### Hypertune ML Model


# In[175]:


### Hyperparameter Tuning or Hyperparameter Optimization
    1.Choose following method for hyperparameter tuning
        a.RandomizedSearchCV --> Fast way to Hypertune model
        b.GridSearchCV--> Slower way to hypertune my model
    2.Choose ML algo that u have to hypertune
    2.Assign hyperparameters in form of dictionary or create hyper-parameter space
    3.define searching &  apply searching on Training data or  Fit the CV model
    4.Check best parameters and best score


# In[176]:


from sklearn.model_selection import RandomizedSearchCV


# In[177]:


### initialise your estimator
reg_rf = RandomForestRegressor()


# In[178]:


np.linspace(start =100 , stop=1200 , num=6)


# In[179]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start =100 , stop=1200 , num=6)]

# Number of features to consider at every split
max_features = ["auto", "sqrt"]

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start =5 , stop=30 , num=4)]

# Minimum number of samples required to split a node
min_samples_split = [5,10,15,100]


# In[180]:


# Create the random grid or hyper-parameter space

random_grid = {
    'n_estimators' : n_estimators ,
    'max_features' : max_features ,
    'max_depth' : max_depth ,
    'min_samples_split' : min_samples_split
}


# In[181]:


random_grid


# In[182]:


## Define searching

# Random search of parameters, using 3 fold cross validation
# search across 576 different combinations


rf_random = RandomizedSearchCV(estimator=reg_rf , param_distributions=random_grid , cv=3 , n_jobs=-1 , verbose=2)


# In[183]:


rf_random.fit(X_train , y_train)


# In[184]:


rf_random.best_params_


# In[185]:


rf_random.best_estimator_


# In[186]:


rf_random.best_score_


# In[ ]:




