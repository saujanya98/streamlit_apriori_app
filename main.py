# Importing the libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from apyori import apriori

# creating the containers i.e. sections for the app

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Welcome to my Apriori Project')
    st.text('The Apriori algorithm refers to the algorithm which is used to calculate the \nassociation rules between objects. It means how two or more objects are \nrelated to one another. In other words, we can say that apriori \nis an association rule learning algorithm that analyzes that people who bought \nproduct A also bought product B. Helpful in scenarios of creating \nbuy-one-get-one-free offers.')

with dataset:
    st.header('Basket of Food Dataset')
    st.text('The dataset displays items that customers bought e.g. pasta, cream \nNote: a row corresponds to items a customer bought.')
    
    dataset = pd.read_csv('data/Market_Basket_Optimisation.csv', header= None)
    st.write(dataset.head())
    transactions =[]
    for i in range(0 ,7501):
        transactions.append([str(dataset.values[i ,j]) for j in range(0, 20)])
        
    st.text('Here are the 10 most frequently bought items:')
    
    transaction = []
    for i in range(0, dataset.shape[0]):
        for j in range(0, dataset.shape[1]):
            transaction.append([ dataset.values[i,j] for j in range(0,1) if dataset.values[i,j]!='0'])
    transaction = np.array(transaction)

    # Transform a pandas dataFrame
    df = pd.DataFrame(transaction, columns=["items"]) 
    df["incident_count"] = 1 # Put 1 to Each Item For Making Countable Table, to be able to perform Group By
    df_table= df.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
    #The 20 most demanded items in dataset eith barplot.
    sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    plt.xticks(rotation=90)
    sns.barplot(x="items",y="incident_count",data=df_table.head(10))
    st.pyplot(fig)
     

with model_training:
    st.header('Training the Apiori model')
    st.text('Using an apriori algorithm, we are trying to find association rules between \ndifferent foods.')
    st.text('Use the sliders to change the hyperparameters of the model')

    min_confidence = st.slider('What should be the minimum confidence of the model?', min_value=0.1, max_value=0.5, value=0.2, step=0.1)
    min_lift = st.slider('What should be the minimum lift of the model?', min_value=2, max_value=6, value=3, step=1)
    
    #Training apriori on the dataset
    rules =apriori(transactions, min_support =0.003, min_confidence =min_confidence, min_lift =min_lift, min_length =2, max_length = 2)

    #Visualising the results
    results =list(rules)

    #Putting the results in a DF
    def inspect(results):
        lhs = [tuple(result[2][0][0])[0] for result in results]
        rhs = [tuple(result[2][0][1])[0] for result in results]
        support = [result[1] for result in results]
        confidence = [result[2][0][2] for result in results]
        lift = [result[2][0][3] for result in results]
        return list(zip(lhs,rhs,support,confidence,lift))

    resultsinDF = pd.DataFrame(inspect(results), columns = ['Product A','Product B','Support','Confidence','Lift'])
    
    st.text('As a result of the parameters chosen to train the apriori model, here are the \nassociation rules returned:')
    st.write(resultsinDF)
    
    
    st.write("Shutdown Media we're back")
    
    
    
    
