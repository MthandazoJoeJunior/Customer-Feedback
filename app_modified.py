from json import dump
from flask import Flask, render_template, request, session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import pandas as pd
from pandas.io.json import json_normalize
import csv
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import webbrowser
from fpdf import FPDF

app = Flask(__name__,template_folder = r"C:\Users\ssd\Downloads\Compressed\text_classification-main\text_classification-main") 

app.secret_key = 'You Will Never Guess'                                                                                   
    
clf = pickle.load(open('clf.pkl','rb'))
loaded_vec = pickle.load(open("count_vect.pkl", "rb"))
declarative_list = []
decla_sent = pd.DataFrame
df2 = pd.DataFrame()
df1 = pd.DataFrame()
df3 = pd.DataFrame()
name='string'
tnum = 0
dnum = 0

def class_scores(data_frame):

   dsent = 0
   insent = 0
   imsent = 0
   exsent = 0

   sentiment_list = []
   for row_num in range(len(data_frame)):
      sentence = data_frame['sentence'][row_num]
      result_pred = clf.predict(loaded_vec.transform([sentence]))
      classification = " "
      for i in result_pred:
        classification += i 
      sentiment_list.append(classification)

      declarative = 'declarative'
      if declarative in classification:
         declarative_list.append([sentence])
         dsent = dsent + 1
      
      interrogative = 'interrogative'
      if interrogative in classification:
         insent = insent + 1

      imperative = 'imperative'
      if imperative in classification:
         imsent = imsent + 1

      exclamatory = 'exclamatory'
      if exclamatory in classification:
         exsent = exsent + 1   

   data_frame['sentence class'] = sentiment_list

   # creating dataframe with declarative sentences
   decla_sent = pd.DataFrame(declarative_list, columns=['declarative_sentences'])
   session['declarative_sentence_file'] = decla_sent.to_json()

   tnum = row_num + 1 #toal number of sentences in a file
   dsentPercent = "{:.2f}".format((dsent / tnum) * 100)
   insentPercent = "{:.2f}".format((insent / tnum) * 100)
   imsentPercent = "{:.2f}".format((imsent / tnum) * 100)
   exsentPercent = "{:.2f}".format((exsent / tnum) * 100)

   df2['Numbers'] = [dsent, insent, imsent, exsent]
   df2['Percentage'] = [dsentPercent, insentPercent, imsentPercent, exsentPercent]
   df1['TotalSent'] = [tnum]

   return data_frame    

def sent_polarity(data_frame):

   pos = 0
   neg = 0
   neu = 0

   SID_obj = SentimentIntensityAnalyzer()
   decla_list = []
   for row_num in range(len(declarative_list)):
      sentence = data_frame['declarative_sentences'][row_num]
      polarity_dict = SID_obj.polarity_scores(sentence)

      row_num += 1
        # Calculate overall sentiment by compound score
      if polarity_dict['compound'] >= 0.05:
        decla_list.append("Positive")
        pos = pos + 1

      elif polarity_dict['compound'] <= - 0.05:
        decla_list.append("Negative")
        neg = neg + 1

      else:
        decla_list.append("Neutral") 
        neu = neu + 1  

   data_frame['polarity'] = decla_list
   dnum = row_num
   posPercent = "{:.2f}".format((pos / dnum) * 100)
   negPercent = "{:.2f}".format((neg / dnum) * 100)
   neuPercent = "{:.2f}".format((neu / dnum) * 100)

   df3['DNumbers'] = [pos, neg, neu]
   df3['DPercentages'] = [posPercent, negPercent, neuPercent]
   df1['TotalDecla'] = [dnum]

   return data_frame 

def generate_report(String):

  df2['Sentences'] = ["Declarative", "Interrogative", "Imperative", "Exclamatory"]
  df3['Polarity'] = ["Positive", "Negative", "Neutral"]

  fig = plt.figure(figsize=(4,1.5))
  ax1 = fig.add_subplot()
  ax1.pie(df2['Percentage'], labels=df2['Sentences'], autopct = '%1.1f%%', textprops={'fontsize':6, 'color':'black'})
  # plt.legend(labels = df2['Sentences'], fontsize = 6, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2)                    
  ax1.axis('equal')
  # plt.title('Sentence Types Chart')
  plt.savefig('sentpiechart.png')    

  ax2 = fig.add_subplot()
  ax2.pie(df3['DPercentages'], labels = df3['Polarity'], autopct = '%1.1f%%', textprops={'fontsize':6, 'color':'black'})
  ax2.axis('equal')
  plt.savefig('declapiechart.png')              

  pdf = FPDF()
  pdf.add_page()
  pdf.set_xy(0, 0)
  pdf.set_font('arial', 'B', 15)
  # pdf.set_text_color(54,65,76)
  pdf.cell(60)
  pdf.cell(90, 10, " ", 0, 2, 'C')
  pdf.cell(75, 10, "Customer Feedback Classification and Analysis Using CNN", 0, 2, 'C')
  pdf.set_font('arial', 'B', 12)
  pdf.cell(75, 10, "Classification and Analysis Report", 0, 2, 'C')
  pdf.cell(90, 10, " ", 0, 2, 'C')
  pdf.cell(-10)
  pdf.cell(75, 10, "Total Number of Customer Feedbacks in Uploaded File : " + str((df1['TotalSent'].iloc[0])), 0, 2, 'C')
  pdf.cell(10)
  pdf.cell(-40)
  pdf.cell(50, 9, 'Sentence Type', 1, 0, 'C')
  pdf.cell(40, 9, 'Number', 1, 0, 'C')
  pdf.cell(40, 9, 'Percentage (%)', 1, 2, 'C')
  pdf.cell(-90)
  pdf.set_font('arial', '', 12)
  for i in range(0, len(df2)):
    pdf.cell(50, 9, '%s' % (df2['Sentences'].iloc[i]), 1, 0, 'C')
    pdf.cell(40, 9, '%s' % (str(df2['Numbers'].iloc[i])), 1, 0, 'C')
    pdf.cell(40, 9, '%s' % (str(df2['Percentage'].iloc[i])), 1, 2, 'C')
    pdf.cell(-90)
  pdf.cell(90, 10, " ", 0, 2, 'C')
  pdf.set_font('arial', 'B', 12)
  pdf.cell(75, 10, "Graphical Presentation", 0, 2, 'C')
  pdf.image('sentpiechart.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
  pdf.cell(40)
  pdf.cell(90, 20, " ", 0, 2, 'C')
  pdf.cell(75, 10, "Number of Declarative Sentences : "  + str((df1['TotalDecla'].iloc[0])), 0, 2, 'L')
  pdf.cell(-40)
  pdf.cell(50, 9, 'Polarity', 1, 0, 'C')
  pdf.cell(40, 9, 'Number', 1, 0, 'C')
  pdf.cell(40, 9, 'Percentage (%)', 1, 2, 'C')
  pdf.cell(-90)
  pdf.set_font('arial', '', 12)
  for i in range(0, len(df3)):
    pdf.cell(50, 9, '%s' % (df3['Polarity'].iloc[i]), 1, 0, 'C')
    pdf.cell(40, 9, '%s' % (str(df3['DNumbers'].iloc[i])), 1, 0, 'C')
    pdf.cell(40, 9, '%s' % (str(df3['DPercentages'].iloc[i])), 1, 2, 'C')
    pdf.cell(-90)
  pdf.cell(90, 10, " ", 0, 2, 'C')
  pdf.set_font('arial', 'B', 12)
  pdf.cell(75, 10, "Graphical Presentation", 0, 2, 'C')
  pdf.image('declapiechart.png', x = None, y = None, w = 0, h = 0, type = '', link = '')
  pdf.cell(40)
  pdf.cell(90, 15, " ", 0, 2, 'C')
  pdf.set_font('arial', 'B', 12)
  pdf.cell(50, 10, "Conclusion", 0, 2, 'R')
  pdf.cell(75, 10, "NB: Conclusion is drawn from analyzing the polarity of the declarative sentences", 0, 2, 'C')
  pdf.set_font('arial', '', 12)
  pdf.cell(75, 10,"-  "+ str(df3['DPercentages'].iloc[0]) + "% of the customers are happy with the product/ service offered", 0, 2, 'C')
  pdf.output('test.pdf', 'F')


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
    # Apply sentiment function to get class score
    uploaded_df_sentiment = class_scores(uploaded_df)
    uploaded_df_html = uploaded_df_sentiment.to_html()
    return render_template('show_data.html', data=uploaded_df_html)


@app.route('/analysis')
def Sent_polarity():
    
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('declarative_sentence_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_analysis = sent_polarity(uploaded_df)
    uploaded_df_html = uploaded_df_analysis.to_html()
    return render_template('show_data2.html', data=uploaded_df_html)

@app.route('/report_generation') 
def Generate_Report():
  generate_report(name)
  return render_template('report_page.html')

@app.route('/open_report')  
def open_report():
  path = 'test.pdf'
  os.system(path)
  return render_template('report_page.html')
 
if __name__=='__main__':
    app.run(debug = True)
