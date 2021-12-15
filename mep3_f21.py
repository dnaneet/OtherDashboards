import numpy as np
import plotly.express as px
import pandas as pd
import dask.dataframe as dd


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
st.sidebar.markdown("This dashboard was created using Python and Streamlit by Aneet Narendranath, PhD.")


# Import CSV file of responses.
#df = pd.read_csv('responses.csv')
#df['comments'] = df['If you have any comments on your selection, please feel free to provide them through this text field.']

#Fillna
#df['comments'] = df['comments'].fillna(0)
#comments = df[df['comments'] != 0]

#Comment polarity and subjectivity calculation

#i_polarity = np.array([])
#i_subjectivity = np.array([])
#for row in comments['comments']:
#  blob = TextBlob(row.lower())
#  i_polarity = np.append(i_polarity, blob.polarity)
#  i_subjectivity = np.append(i_subjectivity, blob.subjectivity)



if selection == "Summary":
    st.markdown("### Greater focus on collaboration") 
    st.markdown("I modified MEP3 to have a greater focus on collaboratively-created/team-created works products.  Individual assignments have formative assessment with multiple possible attempts, and feedback while team assignments have used summative assessment (Bain, 2020). Students have engaged with course content and tested their software skills via individual assignments.  Team assignments relied on individually developed skills.  This approach ensured that the focus was on 'teamwork.' The formative-assessment nature of individual assignments allowed me (instructor) to engage students in constructive dialogue rather than arguments ('where/why did I 'lose' points?'').")  
    st.markdown("###  Design Review") 
    st.markdown("Two design reviews (DR1, DR2) were conducted. The first design review in week-6. During DR-1, teams delivered a 2-minute presentation and demonstration of their models and conformation to engineering standards and codes. The purpose of DR-1 was for the instructional team (instructors, TAs, software helpers) to provide critical feedback to student-teams and for peers to review each other's work.   The second design review was conducted in week-13.  Teams delivered a 8 minute presentation of all their results as a digital poster.  This was a final presentation.")
    st.markdown("Media samples submitted by teams towards their Design Review are available on request.")
    st.markdown("### Mastery Quizzes")
    st.markdown("Students must attain a minimum threshold score (70%) on each MQ before being eligible for the next MQ. Two out of three MQs are complete. MEP3 focuses on Software for design. Time-limited examinations are not a good method for testing software skills. This is because problem-solving with software packages can have solution pathways (menu access, button clicks) that can vary from one person to another. The test should be focused on the accuracy of the solution and not on how quickly a student completes the muscle-mechanics of pressing buttons. Mastery quizzes allowed me to engage with students in constructive dialog.  These quizzes are not perpetually open but have an open and close date like any other assignment and source questions from Canvas question-banks to ensure variety.  These quizzes conform to some of the practices of  'universal design.'")
    st.markdown("### Critical Incident Questionnaire (CIQ)")
    st.markdown("I used a Critical Incident Questionnaire (CIQ) to monitor team activity and team dynamics.  The CIQ (Brookfield, 1995) 'seeks to capture the critical moments, experiences, or 'vivid happenings' that occur in a learning episode for the purpose of informing the class instructor or facilitator about how the learning experience is proceeding.'  The MEP3 CIQ was deployed via a Google form and was voluntary for teams to complete. The outcome was a weekly reflection of their activities by the teams, communicated to the instructor.   Unlike the 'early-term survey' deployed through Canvas, the CIQ represents a weekly evolution of team dynamics, instead of a single data point collected in week-4. I used my prior experience with MEP3 and the CIQ data to recognize challenges and provided positive intervention to teams when needed.  This does not perfectly solve 'team problems' but allows me to preemptively allay them.  I speculate that high-functioning teams are the ones that voluntarily completed this survey.  If this were made mandatory, it would develop really good picture of team dynamics within a classroom.")
if selection == "Feedback from students":   
    st.markdown("## Feedback from students on the 4 new elements deployed")
    st.markdown("I requested feedback on the following 4 new learning support elements that I used in MEP3 this semester, via a single-question Google Form survey.  The question on the survey asked students to rate the importance of the following four elements in ensuring they were productive members of their design team.  The rating scale was 1 thru 5 with 1 = not important while 5 = extremel important.")
    '''
     - Unlimited attempts on quizzes (instead of the traditional “single attempt”).
     - Mastery Quizzes (instead of traditional timed exams).
     - 5-6 member design teams (instead of 3-4 member teams).
     - Two design reviews with DR-1 providing intermediate feedback on your design(instead of a single final presentation).
    '''

    #with st.expander("Evaluation scores: how useful these elements were for student preparation:"):
    #df_scores = pd.read_csv("surve_scores.csv")
    #st.table(df_scores)

    with st.expander("Written comments"):
      st.markdown("47 out of 141 students completed the survey but only 20 out of these 47 provided written comments.  These written comments had their sentiment analysed using Machine Learning functions.  A sentiment value of +1 is 'highly positive' while a sentiment value of -1 is 'highly negative'.")
      #df2 = pd.DataFrame(
      #np.random.randn(50, 20),
      #columns=('col %d' % i for i in range(20)))
      df = pd.read_csv('responses.csv')
      df['comments'] = df['If you have any comments on your selection, please feel free to provide them through this text field.']
      df['comments'] = df['comments'].fillna(0)
      comments = df[df['comments'] != 0]
      #
      #Comment polarity and subjectivity calculation

      i_polarity = np.array([])
      i_subjectivity = np.array([])
      for row in comments['comments']:
        blob = TextBlob(row.lower())
        i_polarity = np.append(i_polarity, blob.polarity)
        i_subjectivity = np.append(i_subjectivity, blob.subjectivity)

      #df["Sentiment of comment"] = i_polarity
      #st.table(df[df["comments"] != 0]["comments"])       
      #st.write(i_polarity) 
      st.markdown("### Survey results")
      df_comments = pd.DataFrame({"Comment" : comments["comments"], "Sentiment of comment": i_polarity})
      st.table(df_comments)

    with st.expander("Survey scores -- how important these elements were as perceived by students"):
      st.markdown("Each column holds the scores for one of the learning support elements.  Each row tallies the number of students that assigned this learning support element a usefulness score.")
      df_scores = pd.read_csv("survey_scores.csv") 
      st.table(df_scores)

if selection == "Grade distribution":
  st.markdown('## MEP3 Historical Gradebook Dashboard')
  st.write("This dashboard summarizes student grades in semesters when Dr. Narendranath was one of the instructors of MEP3.")
  st.write(" ")  
  st.markdown("Select a semester and set a threshold score.  The three columns below will return the enrollment, number of students with an overall score > selected threshold, and number of 'F's.")
  sem = st.selectbox('Select semester of interest:', ['Fall 2017', 'Spring 2018', 'Fall 2018', 'Spring 2019', 'Fall 2019', 'Fall 2020', 'Spring 2021', 'Fall 2021'])
  ts =  st.slider('Select threshold final score:', max_value = 100, min_value=0)
  df_grades = pd.read_csv("all_grades.csv")

  col1, col2, col3 = st.columns(3)
  with col1:  
    st.write("Number of students enrolled", df_grades[df_grades["Semester"] == sem]["Semester"].count())
  #NumStudentsGreaterThanTS = df_grades[(df_grades["Semester"] == sem) && (df_grades["Final Score"] > ts)]
  #st.write("Percentage of students who received more than the threshold score:", np.round(100*NumStudentsGreaterThanTS/df_grades[df_grades["Semester"] == sem]["Semester"].count()))
  with col2:
    st.write("Number of students who had a final score greater than the selected threshold:", df_grades[(df_grades["Semester"] == sem) & (df_grades["Final Score"] > ts)]["Final Score"].count())
  with col3:
    st.write("Number of 'F' grades:", df_grades[(df_grades["Semester"] == sem) & (df_grades["Final Grade"] == "F")]["Final Grade"].count())