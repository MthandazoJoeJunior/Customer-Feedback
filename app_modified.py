from json import dump
from flask import Flask, render_template, request, session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import pandas as pd
from pandas.io.json import json_normalize
import csv
import os

app = Flask(__name__,template_folder = r"C:\Users\ssd\Downloads\Compressed\text_classification-main\text_classification-main") 

app.secret_key = 'You Will Never Guess'                                                                                   
    
clf = pickle.load(open('clf.pkl','rb'))
loaded_vec = pickle.load(open("count_vect.pkl", "rb"))
declarative_list = []
df = pd.DataFrame

def class_scores(data_frame):

   sentiment_list = []
   for row_num in range(len(data_frame)):
      sentence = data_frame['sentence'][row_num]

      #result = request.form['uploaded-file']
      result_pred = clf.predict(loaded_vec.transform([sentence]))
      classification = " "
      for i in result_pred:
        classification += i 
      sentiment_list.append(classification)

      declarative = 'declarative'
      if declarative in classification:
         declarative_list.append([sentence])
      

   data_frame['sentence class'] = sentiment_list
   print(declarative_list)

   return data_frame    

def sent_polarity(data_frame):
   SID_obj = SentimentIntensityAnalyzer()

#    declarative_list = []
   df = pd.DataFrame(declarative_list, columns=['declarative_sentences'])
   for row_num in range(len(data_frame)):
      sentence = data_frame['sentence'][row_num]
      polarity_dict = SID_obj.polarity_scores(declarative_list)

        # Calculate overall sentiment by compound score
      if polarity_dict['compound'] >= 0.05:
        declarative_list.append("Positive")

      elif polarity_dict['compound'] <= - 0.05:
        declarative_list.append("Negative")

      else:
        declarative_list.append("Neutral")

   data_frame['polarity'] = declarative_list  

   return data_frame 


@app.route('/')
def index():
    return render_template('index_upload_and_show_data.html')


@app.route('/',methods = ['POST', 'GET'])
def uploadFile():
    if request.method == 'POST':
        uploaded_file = request.files['uploaded-file']
        df = pd.read_csv(uploaded_file)
        session['uploaded_csv_file'] = df.to_json()
        return render_template('index_upload_and_show_data_page2.html')

@app.route('/show_data')
def showData():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Convert dataframe to html format
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_data.html', data=uploaded_df_html)
 
@app.route('/sentiment')
def SentimentAnalysis():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = class_scores(uploaded_df)
    uploaded_df_html = uploaded_df_sentiment.to_html()
    # uploaded_df_analysis = sent_polarity(uploaded_df)
    # uploaded_df_html = uploaded_df_analysis.to_html()
    return render_template('show_data.html', data=uploaded_df_html)


@app.route('/analysis')
def Sent_polarity():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(uploaded_json)
    # Apply sentiment function to get sentiment score
    uploaded_df_analysis = sent_polarity(uploaded_df)
    uploaded_df_html = uploaded_df_analysis.to_html()
    return render_template('show_data.html', data=uploaded_df_html)
 
if __name__=='__main__':
    app.run(debug = True)
