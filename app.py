import streamlit as st
import joblib
from functions import *
import pandas as pd
import docx2txt

lr_model = joblib.load('Logistic.pkl')
dtc = joblib.load( 'DecisionTree.pkl')
rfc = joblib.load('RandomForest.pkl')
abc = joblib.load('AdaBoostClassifier.pkl')
xgb = joblib.load('XGBClassifier.pkl')
knc= joblib.load('KNeighbors.pkl')
nv_model= joblib.load('naive_bayes.pkl')
gbc= joblib.load('Gradientboost.pkl')

st.write("""
# Emotion Detection
""")
st.write("""We have many type of emotions .In that case we can detect only these types of emotion:  Neutral, Joy, Sadness, Fear, Surprise, Anger, Shame,  Disgust.""")

inp=st.radio('Choose',('Input text','Input text file'))

if inp=='Input text':
    text =st.text_input('Input your text')

#file=st.file_uploader('Upload text file')
#st.selectbox(label, options, index=0)
    model = st.selectbox(
        'Select your desire model for detect emotion of your text',
        options=['Logistic', 'DecisionTree', 'RandomForest', 'AdaBoostClassifier', 'XGBClassifier', 'KNeighbors', 'naive_bayes', 'Gradientboost'],
    )
    detect=st.button('Detect')
    compare=st.button('Compare All')

    index={'Logistic':0, 'DecisionTree':1, 'RandomForest':2, 'AdaBoostClassifier':3, 'XGBClassifier':4, 'KNeighbors':5, 'naive_bayes':6, 'Gradientboost':7}
    model_list= [lr_model,dtc,rfc,abc,xgb,knc,nv_model,gbc]

    if detect:
        i=index[model]
        a = PredEmotion([text], model_list[i])
        st.write(a[0])
    if compare:
        Emotion = []
        for i in model_list:
            a = PredEmotion([text], i)
            Emotion.append(a)
        df = {'CLASSIFIER': ['Logistic','DecisionTree', 'RandomForest', 'AdaBoostClassifier', 'XGBClassifier', 'KNeighbors', 'naive_bayes', 'Gradientboost'],

            'Emotion': Extract_1(Emotion),
            'Prediction Score': Extract_2(Emotion)}
        table = pd.DataFrame(df)

        st.write(table)
        Emotion=[]
if inp=='Input text file':
    file =st.file_uploader('Upload your text file',type=['txt','.docx'])
    if str(file.name).endswith('.txt'):
        a_file = open(str(file.name), "r")
        lines = a_file.read()
        list_of_lists = lines.splitlines()
        a_file.close()
    if str(file.name).endswith('.docx'):
        list_of_lists=list(docx2txt.process(str(file.name)))

    doc=''.join(list_of_lists)
    plot_wordcloud(doc)

    model = st.selectbox(
        'Select your desire model for detect emotion of your text',
        options=['Logistic', 'DecisionTree', 'RandomForest', 'AdaBoostClassifier', 'XGBClassifier', 'KNeighbors',
                 'naive_bayes', 'Gradientboost'],
    )
    detect = st.button('Detect')
    compare = st.button('Compare All')

    index = {'Logistic': 0, 'DecisionTree': 1, 'RandomForest': 2, 'AdaBoostClassifier': 3, 'XGBClassifier': 4,
             'KNeighbors': 5, 'naive_bayes': 6, 'Gradientboost': 7}
    model_list = [lr_model, dtc, rfc, abc, xgb, knc, nv_model, gbc]

    if detect:
        i = index[model]
        a = PredEmotion(list_of_lists, model_list[i])
        st.write(a[0])
    if compare:
        Emotion = []
        for i in model_list:
            a = PredEmotion(list_of_lists, i)
            Emotion.append(a)
        df = {'CLASSIFIER': ['Logistic', 'DecisionTree', 'RandomForest', 'AdaBoostClassifier', 'XGBClassifier',
                             'KNeighbors', 'naive_bayes', 'Gradientboost'],

              'Emotion': Extract_1(Emotion),
              'Prediction Score': Extract_2(Emotion)}
        table = pd.DataFrame(df)

        st.write(table)
        Emotion = []