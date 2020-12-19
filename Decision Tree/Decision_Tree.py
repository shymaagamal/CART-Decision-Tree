# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Import Important Package

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics 

# %% [markdown]
# # Load and Prepare Data

# %%
data=pd.read_csv('cardio_train.csv')
data.head()

# %% [markdown]
# ### *The age is given in days, I have to convert it into years*

# %%
data["age"] = data["age"]/365
data["age"] = data["age"].astype("int")

# %% [markdown]
# ### *I don't use id column so i drop it*

# %%
data = data.drop('id',axis=1)
data.rename(columns={'ap_hi':'systolic_bp','ap_lo':'diastolic_bp'},inplace=True)
data.head(1)

# %% [markdown]
# ### *See if have any dublicates in data*

# %%
data.duplicated().sum()

# %% [markdown]
# ### *Drop dublicates*

# %%
data.drop_duplicates(inplace=True)


# %%
data.shape

# %% [markdown]
# ### *See if there is any outliers in data*

# %%
outlier = ((data["systolic_bp"]>200) | (data["diastolic_bp"]>180) | (data["diastolic_bp"]<50) | (data["systolic_bp"]<=80) )
print("There is {} outlier".format(data[outlier]["cardio"].count()))

# %% [markdown]
# ### *Drop outliers*

# %%
# Removing  the outlier from the Dataset.
data = data[~outlier]

# %% [markdown]
# ### *Height and weight seems uncorrelated with the cardio feature but Body Mass Index (BMI) could be more helpful*
# ### *i use pulse pressure to determine cardio feature to reduce numbers of feature*

# %%
data["bmi"] = data["weight"]/ (data["height"]/100)**2


# %%
data = data.drop(['weight','height'],axis=1)


# %%
data["gender"] = data["gender"] % 2
data.head(5)

# %% [markdown]
# ### *BMI between  18.5 and 25 , personis Normal*
# ### *if BMI obove 25 , person is obese*
# ### *if BMI less than 18.5 , person is underweight *

# %%

for i,row in data.iterrows():
    if row['bmi'] <18.5 :
        data.at[i,'bmi'] = 0
    elif row['bmi'] >25 :
        data.at[i,'bmi'] = 2
    else  :
        data.at[i,'bmi'] = 1      

# %% [markdown]
# ### *systolic blood pressure number:*
# 
# *Normal: Below 120*
# 
# *Elevated: 120-129*
# 
# *Stage 1 high blood pressure (also called hypertension): 130-139*
# 
# *Stage 2 hypertension: 140 or more*
# 
# *Hypertensive crisis: 180 or more.*

# %%
for i,row in data.iterrows():
    if row['systolic_bp'] <120 :
        data.at[i,'systolic_bp'] = 0
    elif (row['systolic_bp'] >=120) and (row['systolic_bp'] <=129) :
        data.at[i,'systolic_bp'] = 1
    elif (row['systolic_bp'] >=130) and (row['systolic_bp'] <=139) :
        data.at[i,'systolic_bp'] = 2
    elif (row['systolic_bp'] >=140) and (row['systolic_bp'] <179) :
        data.at[i,'systolic_bp'] = 3
    else  :
        data.at[i,'systolic_bp'] = 4  

# %% [markdown]
# ### *diastolic blood pressure number means:*
# 
# *Normal: Lower than 80*
# 
# *Stage 1 hypertension: 80-89*
# 
# *Stage 2 hypertension: 90 or more*
# 
# *Hypertensive crisis: 120 or more*

# %%
for i,row in data.iterrows():
    if row['diastolic_bp'] <80 :
        data.at[i,'diastolic_bp'] = 0
    elif (row['diastolic_bp'] >=80) and (row['diastolic_bp'] <=90) :
        data.at[i,'diastolic_bp'] = 1
    elif (row['diastolic_bp'] >90) and (row['diastolic_bp'] <=120) :
        data.at[i,'diastolic_bp'] = 2
    else  :
        data.at[i,'diastolic_bp'] = 3  


# %%
data.age.unique()


# %%
for i,row in data.iterrows():
    if (row['age'] >=20) and (row['age'] <=40) :
        data.at[i,'age'] = 0
    elif (row['age'] >40) and (row['age'] <=45) :
        data.at[i,'age'] = 1
    elif (row['age'] >45) and (row['age'] <=50) :
        data.at[i,'age'] = 2
    elif (row['age'] >50) and (row['age'] <=55) :
        data.at[i,'age'] = 3
    elif (row['age'] >55) and (row['age'] <=60) :
        data.at[i,'age'] = 4
    else  :
        data.at[i,'age'] = 5

# %% [markdown]
# ### *Rearrange Columns*

# %%
data =  data[ [ col for col in data.columns if col != 'cardio' ]+['cardio'] ]


# %%
data.head(3)

# %% [markdown]
# # Train and Test Split

# %%
def train_test_splits(df,test_size):
    if isinstance(test_size,float):
        test_size=round(test_size *len(df))
    indcies=df.index.tolist()
    test_indices=random.sample(population=indcies,k=test_size)
    test_set=df.loc[test_indices]
    train_set=df.drop(test_indices)
    return train_set,test_set


# %%
random.seed(0)
train_set,test_set=train_test_splits(data,test_size=0.1)

# %% [markdown]
# Algorithm
# ![](Algorithm.png)
#  

# %%
def check_purity(data): 
    labels = data[:,-1]
    unique_classes = np.unique(labels)

    if len(unique_classes) == 1:
        return True
    else:
        return False


# %%
check_purity(train_set.values)

# %% [markdown]
# ### *Return Majority Class*

# %%
def classify_data(dataset): 
    labels = dataset[:,-1]
    unique_classes, count_unique_classes = np.unique(labels, return_counts=True)
    index = count_unique_classes.argmax()
    classification = unique_classes[index]
    return classification


# %%
classify_data(train_set[train_set.age<30].values)

# %% [markdown]
# ### *Potential_splits*

# %%
def get_potential_split(data):
    potential_splits = {}
    n_cols = data.shape[1]  # Number of columns
    for i_col in range(n_cols - 1): # Disregarding the last label column  
        potential_splits[i_col] = []
        values = data[:,i_col]
        unique_values = np.unique(values)   # All possible values
        for index in range(len(unique_values)):
            if index !=0 :
                current_value=unique_values[index]
                previous_value=unique_values[index-1]
                potential_splits[i_col].append((current_value+previous_value)/2)
    return potential_splits


# %%
get_potential_split(train_set.values)

# %% [markdown]
# ### *Split data*

# %%
def split_data(data,split_column, split_value):
    split_column_values = data[:, split_column]

    left = data[split_column_values <= split_value]
    right = data[split_column_values >  split_value]

    return left, right


# %%
left,right=split_data(train_set.values,3,80)

# %% [markdown]
# ### Lowest Overall gini
# 
# ### Gini impurity 
# ## *$G=1-\sum^{n}_{k=1}{p_{k}^{2}}$*

# %%
def calculate_gini(data): 
    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()
    gini = 1 - sum(np.square(probs))

    return gini

# %% [markdown]
# ### Cost Function That is minimized in classification
# 
# ### Overall Gini
# ## *$J=\frac{m_{left}}{m} G_{left}+\frac{m_{right}}{m} G_{right}$*

# %%
def calculate_overall_gini(left, right): 
    total_num = len(left) + len(right)
    prob_left = len(left) / total_num
    prob_right = len(right) / total_num

    overall_gini = prob_left * calculate_gini(left) + prob_right * calculate_gini(right)

    return overall_gini 


# %%
calculate_overall_gini(left,right)

# %% [markdown]
# ### *Find best feature and best valur to this feature to split data*

# %%
def find_best_split(data, potential_splits): 
    global best_split_column, best_split_value

    min_overall_impurity = float('inf') # Store the largest overall impurity value
    for coulmn_index in potential_splits:
        for value in potential_splits[coulmn_index]:
            left,right = split_data(data,coulmn_index, value)
            overall_impurity = calculate_overall_gini(left, right)

            if overall_impurity <= min_overall_impurity:    # Find new minimised impurity
                min_overall_impurity = overall_impurity     # Replace the minimum impurity
                best_split_column = coulmn_index
                best_split_value = value
    return best_split_column, best_split_value


# %%

splits=get_potential_split(train_set.values)

find_best_split(train_set.values,splits)

# %% [markdown]
# ## Decision Tree Algorithm

# %%
def decision_tree_algorithm(data,counter=0,min_sample=10,max_depth=6):
    #data Preperation 
    if counter==0:
        global COLUMN_HEADERS
        COLUMN_HEADERS=data.columns
        data=data.values
        

    else:
        
        data=data 
    # base Algorithm ==> recursive function
    if (check_purity(data)) or (len(data)< min_sample) or(counter==max_depth) :
        classification=classify_data(data)
        return classification
        #recursive part   
    else:
        counter+=1
        #helper function 
        potential_splits=get_potential_split(data)

        split_column,split_value=find_best_split(data,potential_splits)
        left,right=split_data(data,split_column,split_value)

        #instant sub tree
        features_name=COLUMN_HEADERS[split_column]
        question ="{} <= {}".format(features_name,split_value)
        sub_tree={question:[]}


        #find answer(recuresion)
        yes_answer=decision_tree_algorithm(left,counter,min_sample,max_depth)
        no_answer=decision_tree_algorithm(right,counter,min_sample,max_depth)
        if yes_answer==no_answer:
            sub_tree=yes_answer

        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree


# %%
tree=decision_tree_algorithm(train_set,min_sample=10)

# %% [markdown]
# ## Classification
# %% [markdown]
# ### *Classifiy Just one instance*

# %%
question=list(tree.keys())[0]
question


# %%
def classify_instance(instance,tree):        
        question=list(tree.keys())[0]
        feature_name,comparison_operator,value = question.split()

        #ask question 
        if instance[feature_name] <= float(value):
            answer=tree[question][0]
        else:
            answer=tree[question][1]

        #base case
        if not isinstance(answer,dict):
            return answer
        else:
            residual_tree = answer
            return classify_instance(instance,residual_tree)   
                


# %% [markdown]
# ### *Classifiy all instances from test set*

# %%
def predict(test_set,tree):
    predications=list()

    for i in range(test_set.shape[0]):
        predications.append(classify_instance(test_set.iloc[i],tree))

    return predications


# %%
Y_p=predict(test_set,tree)

# %% [markdown]
# ## Evaluate Performance

# %%
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 


# %%
actual=test_set.iloc[:,-1]
accuracy_metric(actual.values,Y_p)

# %% [markdown]
# ## Use My Model 

# %%
random.seed(0)
train_df,test_df=train_test_splits(data,test_size=0.1)
Tree=decision_tree_algorithm(train_df,min_sample=10)
y_predict=predict(test_set,Tree)
actual=test_df.iloc[:,-1]
accuracy_metric(actual.values,y_predict)

# %% [markdown]
# # Use Sicit_Learn Model To compare score

# %%
col= data.shape[1]
X= data.iloc[:,:col-1]
Y=data.iloc[:,col-1:col]


# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# %%
Model = DecisionTreeClassifier()
Model.fit(X_train,y_train)
y_pred = Model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

# %% [markdown]
# # Bagging Ensemble Learning.

# %%
def Bagged_fitting(data, num_of_bagged):
        models=[]
       
        for i in range(num_of_bagged):
            sample=data.sample(n=len(data))
            model=decision_tree_algorithm(sample)
            models.append(model)
        return models   


# %%
def Bagged_prdiction(test_set,models,num_of_bagged):
        pred=np.zeros(len(test_set))
        for model in models:
            pred+=predict(test_set,model)  
        return np.round(pred/num_of_bagged) 

# %% [markdown]
# # Use My Bagging Ensemble Model

# %%
trees=Bagged(train_set,9)


# %%
actual=test_set.iloc[:,-1]
YM=Bagged_prdiction(test_set,trees,9)
accuracy_metric(actual.values,YM)


# %%



