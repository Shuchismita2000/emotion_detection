from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import joblib
from wordcloud import WordCloud
import  matplotlib.pyplot as plt
cv = joblib.load('vectorizer.pkl')
def PredEmotion(sample_text,model):
    myvect=cv.transform(sample_text).toarray()
    prediction=model.predict(myvect)
    prediction_probability=model.predict_proba(myvect)
    max_proba=np.max(prediction_probability)
    pred=prediction[0]
    l=[pred,max_proba]
    return l

def Extract_1(lst):
    return [item[0] for item in lst]
def Extract_2(lst):
    return [item[1] for item in lst]

def plot_wordcloud(docx):
    mywordcloud=WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()
