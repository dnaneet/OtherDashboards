import numpy as np
import plotly.express as px
import pandas as pd


import nltk # https://www.nltk.org/install.html
from nltk.tokenize import word_tokenize
from nltk.text import Text

#import spacy
#from spacy import displacy
#from collections import Counter
#import en_core_web_sm
#nlp = en_core_web_sm.load()
#from pprint import pprint

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download([ "names", "stopwords", "state_union", "twitter_samples", "movie_reviews", "averaged_perceptron_tagger", "vader_lexicon", "punkt"])
nltk.download('maxent_ne_chunker')
nltk.download('words')
stopwords = nltk.corpus.stopwords.words("english")

import streamlit as st


st.set_page_config(
     page_title='MEP3 Dashboard Fall 2021'
     #layout="wide"
)
selection = st.sidebar.radio('Select ', ["Summary", "Feedback from students", "Grade distribution"])


# Import CSV file of responses.
df = pd.read_csv('responses.csv')
df['comments'] = df['If you have any comments on your selection, please feel free to provide them through this text field.']

#Fillna
df['comments'] = df['comments'].fillna(0)
comments = df[df['comments'] != 0]

#Comment polarity and subjectivity calculation

i_polarity = np.array([])
i_subjectivity = np.array([])
for row in comments['comments']:
  blob = TextBlob(row.lower())
  i_polarity = np.append(i_polarity, blob.polarity)
  i_subjectivity = np.append(i_subjectivity, blob.subjectivity)



if selection == "Summary":
    st.markdown("### Greater focus on collaboration") 
    st.markdown("I modified MEP3 to have a greater focus on collaboratively-created/team-created works products.  Individual assignments have formative assessment with multiple possible attempts, and feedback while team assignments have used summative assessment (Bain, 2020). Students have engaged with course content and tested their software skills via individual assignments.  Team assignments relied on individually developed skills.  This approach ensured that the focus was on 'teamwork.' The formative-assessment nature of individual assignments allowed me (instructor) to engage students in constructive dialogue rather than arguments ('where/why did I 'lose' points?'').")  
    st.markdown("###  Design Review") 
    st.markdown("Two design reviews (DR1, DR2) were conducted. The first design review in week-6. During DR-1, teams delivered a 2-minute presentation and demonstration of their models and conformation to engineering standards and codes. The purpose of DR-1 was for the instructional team (instructors, TAs, software helpers) to provide critical feedback to student-teams and for peers to review each other's work.   The second design review was conducted in week-13.  Teams delivered a 8 minute presentation of all their results as a digital poster.  This was a final presentation.")
if selection == "Feedback from students":
    st.markdown("TBA")
if selection == "Grade distribution":
    st.markdown("TBA")    
