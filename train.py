import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from  sklearn.feature_extraction.text  import CountVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
import nltk 
import  re 
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer

# from  the vast text corpus ( body of large text ) import the stop words such as [ is   the , am , that , ] 
# i.e. the wprds that do not carry a meaning individually 
# 
nltk.download( 'stopwords')
stopword = set( stopwords.words('english')) # make a set of these words by getting the stopwords from the english langage 
stemmer = SnowballStemmer( "english") # stemmer means to change the words to their intial or actual ( first form )  
# eg . hating -> hate , loveing -> love so that awe can proccess only the meaing that the prds carry and leave out the unecessary 
#letters inthe processing . 

 # exaple ( cause i am doing this for the first time )
# Sample text

# %%
sentence = "They are hating on others in the worst doing shaking baking lovely sun you don't know me son knowing  way possible!"

# Preprocessing
words = sentence.lower().split()
answer=[]
for i in words : 
    if i  not in stopword : 
        answer.append( stemmer.stem(i))
        
print( answer )

# Set the working directory
os.chdir("E:\DATA\PROJECTS\HATE SPEECH RECOGNITION")

# Load the CSV file into a DataFrame
data = pd.read_csv("labeled_data.csv")
#%%
## CATEGORIZATION OF COMMENTS AS HATE, OFFENSIVE OR NONE
data['labels'] = data['class'].map({0 : 'hate speech'  , 1 :'offensive language'  , 2 : 'neither hate nor offensive'} ) 
data_cliff  = data[ [ 'tweet'  ,'labels' ]]
data_cliff
# %%
##  NLP BEGINS  (text processing , cleaning )
def cleaning( txt) : 
    txt= str(txt).lower() 
    txt  = re.sub( '[.?]' , '' , txt)
    txt = re.sub( 'https?://\S|www.\S+' , '' , txt) 
    txt = re.sub( '<.?>+' , '' , txt)
    txt = re.sub(r'[^\w\s]','',txt)
    txt = re.sub( '\n' , '' , txt) 
    txt = re.sub('\w\d\w' ,'' , txt) 
    txt = [i for i in txt.split(' ') if i not  in stopword]
    txt = " ".join(txt) 
    txt = [stemmer.stem(word)  for  word in txt.split(' ')]
    txt = " ".join(txt)
    return txt 



test_txt = "Hey there! 👋 Did you check out our new product a1b? It's amazing!!! Visit https://www.example.com now. <br>Also, check www.testsite.org for updates. New line starts here.\nAnd this\tline has a tab!Here's some junk: #$%^&*()_+=~`<>?/Don't forget to RSVP by 9pm. Thank you!"


cleaning( test_txt)
# %%
data["cleaned tweet"] = data['tweet'].apply(cleaning)
data 
# DATA MODELING PREP 
x = np.array( data['cleaned tweet'])
y = np.array( data['labels'])
print(x)
print(y)

## JOINING TOKENIZED WORDS FOR TEXT PROCESSING AND TRAINING IN MODEL
# no need ofr this because the x and y array are already in the form of  joined sscentences . 

# %%
## INITIALIZING COUNT VECTORIZER (TEXT --> NUMERICAL REPRESENTATION BASED ON WORD COUNTS FOR TRAINING)

#cv = CountVectorizer()
#x = cv.fit_transform(x) 

xtrn,xtst,ytrn,ytst = train_test_split( x, y , test_size= 0.33 , random_state= 42 )

from sklearn.ensemble import RandomForestClassifier

rfclass = RandomForestClassifier(n_estimators=100, random_state=42)
rfclass.fit(xtrn, ytrn)


# %%

# PREDICTION

ypred  = rfclass.predict( xtst) 

from sklearn.metrics import accuracy_score 
print( accuracy_score( ytst  , ypred ) ) 

accuracy = ( accuracy_score( ytst  , ypred ) )*100 
print( " your accuracy off  descision treee classification model is :" , accuracy )
# example to test
kaka = " you dumb ass bitch ,  i will kill you with a sledge hammer and  then play around with your blood , actually i am into murders and executions "
kaka = cv.transform( [kaka]).toarray()
print( rfclass.predict( (kaka)))


# %%
# for understanding how the feature names i.e. the  column names are  visible 

words = cv.get_feature_names_out()
counts = kaka[0]

for word, count in zip(words, counts):
    if count > 0:
        print(f"{word}: {count}")

# %%
## for a  USER 
user_input = input("Enter your comment: ")

user_text_1 = cv.transform([user_input]).toarray()

result = rfclass.predict(user_text_1)[0]  # Get the string, not ['string']

print("\nYou typed:", user_input)
print("Prediction:", result)

# Define your response function
def toretto(result):
    print("\n--- RESPONSE ---")
    if result == "hate speech":
        print("Category:", result)
        print("“In times of crisis, the wise build bridges while the foolish build barriers.” – T’Challa, Black Panther")
        print("Hate divides, but understanding unites. Let your words be a bridge, not a barrier.")


    elif result == "offensive language":
        print("Category:", result)
        print("The best messages are those that inspire, not offend.")
        print("“Words are, in my not-so-humble opinion, our most inexhaustible source of magic.” – Albus Dumbledore")
        print("They can inflict harm, or they can heal. Choose to be the magic that heals.")


    elif result == "neither hate nor offensive":
        print("Category:", result)
        print("“With great power, there must also come great responsibility.” – Aunt May, Spider-Man: No Way Home")
        print("Your words carry power — and you're using them wisely. Keep it up!")

    else:
        print("Unknown category. Please check the model or input.")

# Call the response function
toretto(result)

# %%

import joblib
# %%
# save the  trained model 

#joblib.dump(rfclass , "model/model.pkl")
# %%
import os
import joblib

# Create the directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the model
joblib.dump(rfclass, "model/model.pkl")
print("Model saved successfully!")

# %%
joblib.dump(cv, "model/vectorizer.pkl")


# %%
'''import os

print("Current working directory:")
print(os.getcwd())

# %%
import os

# Path to the directory you want to switch to
new_directory = r"E:\DATA ANLYST\PROJECTS\HATE SPEECH RECOGNITION\STREAMLIT_APP"

# Change directory
os.chdir(new_directory)

# Verify
print("Changed to:", os.getcwd())
'''
# %%
