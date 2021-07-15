import numpy as np
import pandas as pd
#import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
     page_title='Streamlit cheat sheet',
     layout="wide",
     initial_sidebar_state="expanded",
)

st.sidebar.write("You can control the amplitude of Signal 2.")
Amplitude = st.sidebar.slider('Choose an amplitude for the second signal:', min_value = 1.5, max_value = 5.0)
st.sidebar.markdown("---")
NormalizationMethod = st.sidebar.radio('Method of Normalization:',["Basic", "MinMax", "Standardization"])

st.title("A Primer on Normalizing Data")

#col1, col2 = st.beta_columns(2)

t = np.linspace(0, 2*np.pi)
signal1 = np.sin(t)
signal2 = Amplitude*np.sin(t)


st.write(""" We have two signals, viz., Signal 1 and Signal 2.  They perhaps measure the same phenomenon but one of the signals has a *gain* """)
st.write("Through choice of Method of Normalization radio buttons, explore the three different types of data normalization.  In each case, pay close attention to how the data changes after being normalized and what the maximum and minimum limits of X and Y axes are for the normalized data.")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t, y=signal1,
    name='signal1',
    mode='markers',
    marker_color='rgba(152, 0, 0, .8)'
))
fig.add_trace(go.Scatter(
    x=t, y=signal2,
    name='signal2',
    mode='markers',
    marker_color='rgba(2, 80, 105, .8)'
))
fig.update_traces(mode='markers', marker_line_width=2, marker_size=13)
st.plotly_chart(fig)




if NormalizationMethod == "Basic":
    st.write(""" ## Signal 1 and Signal 2 normalized by their maximum amplitude""")
    st.write("When each signal is normalized (or *scaled*) using their maximum amplitude, they are easier to compare with each other.")
    st.latex(r''' \frac{\text{data}}{max(\text{data})} ''')
    signal1_n1 = signal1/max(signal1);
    signal2_n1 = signal2/max(signal2);
    #df['signal1n1'] = signal1_n1;    #df['signal2n1'] = signal2_n1;
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal1_n1,
        name='signal1 (normalized)',
        mode='markers',
        marker_color='rgba(152, 0, 0, .8)'
    ))
    fig.add_trace(go.Scatter(
        x=t, y=signal2_n1,
        name='signal2 (normalized)',
        mode='markers',
        marker_color='rgba(2, 80, 105, .8)'
    ))
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=13)
    st.plotly_chart(fig)    
elif NormalizationMethod == "MinMax":
    st.write("## Signal 1 and Signal 2 are 'MinMax Normalized' ")
    st.write("MinMax normalization is yet another method of scaling two (or more) signals.  In this case the normalization is achieved through the operation:")
    st.latex(r''' \frac{\text{data} - min(\text{data})}{max(\text{data}) - min(\text{data})} ''')
    signal1_n2 = (signal1 - min(signal1))/(max(signal1) - min(signal1));
    signal2_n2 = (signal2 - min(signal2))/(max(signal2) - min(signal2));
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal1_n2,
        name='Signal 1 (MinMax-ed)',
        mode='markers',
        marker_color='rgba(152, 0, 0, .8)'
    ))
    fig.add_trace(go.Scatter(
        x=t, y=signal2_n2,
        name='Signal 2 (MinMax-ed)',
        mode='markers',
        marker_color='rgba(2, 80, 105, .8)'
    ))
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=13)
    st.plotly_chart(fig)
else:
    st.write("## Normalization of the signals through *Standardization*")
    st.write("Standardization is achieved through the operation:")
    st.latex(r''' \frac{\text{data} -  mean(\text{data})}{StdDev(\text{data})} ''')
    signal1_n3 = (signal1 - np.mean(signal1))/np.std(signal1)
    signal2_n3 = (signal2 - np.mean(signal2))/np.std(signal2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal1_n3,
        name='Signal 1 (standarized)',
        mode='markers',
        marker_color='rgba(152, 0, 0, .8)'
    ))
    fig.add_trace(go.Scatter(
        x=t, y=signal2_n3,
        name='Signal 2 (standarized)',
        mode='markers',
        marker_color='rgba(2, 80, 105, .8)'
    ))
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=13)
    st.plotly_chart(fig)


st.write("###### Created by Aneet Narendranath, PhD")    