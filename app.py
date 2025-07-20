import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

pipe1 = pickle.load(open('t20i.pkl','rb'))
pipe2 = pickle.load(open('odi.pkl','rb'))

teams = ['Australia','India','Bangladesh','New Zealand','South Africa',
         'England','West Indies','Pakistan','Sri Lanka']

cities = ['Colombo','Mirpur','Johannesburg','Dubai','Auckland','Cape Town',
          'London','Pallekele','Barbados','Sydney','Melbourne','Durban',
          'St Lucia','Wellington','Lauderhill','Hamilton','Centurion',
          'Manchester','Abu Dhabi','Mumbai','Nottingham','Southampton',
          'Mount Maunganui','Chittagong','Kolkata','Lahore','Delhi',
          'Nagpur','Chandigarh','Adelaide','Bangalore','St Kitts',
          'Cardiff','Christchurch','Trinidad']

m_format = ['ODI','T20i']


st.title('Score Predictor')

col1, col2, col3 = st.columns(3)

with col1:
    batting_team = st.selectbox('Select Batting Team:',sorted(teams))

with col2:
    bowling_team = st.selectbox('Select Bowling Team:',sorted(teams))

with col3:
    sm_format = st.selectbox('Select Format:',sorted(m_format))

city = st.selectbox('Select City: ',sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done(works for over>5)')
with col5:
    wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    
    wickets_left = 10 -wickets
    crr = current_score/overs

    if sm_format == 'T20i':
        balls_left = 120 - (overs*6)
        input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
        result = pipe1.predict(input_df)
    elif sm_format == 'ODI':
        balls_left = 300 - (overs*6)
        input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
        result = pipe2.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))