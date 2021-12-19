#import
import pymysql
from sqlalchemy import create_engine
import MySQLdb
from konlpy.tag import Okt 
okt = Okt()
import numpy as np
import pandas as pd
import os
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from pandas import Series, DataFrame
import nltk


#sql 연결 코드

host = 'database-skku.c6dzc5dnqf69.ap-northeast-2.rds.amazonaws.com'
iid ='admin'
pw = 'tjdrbsrhkseo123'
db_name = 'dongwan'
conn = pymysql.connect(host=host, user= iid, password=pw, db=db_name, charset='utf8')

curs = conn.cursor(pymysql.cursors.DictCursor)

sql1 = """SELECT * FROM public.clinical_disease"""
sql2 = """SELECT * FROM medii.TotalDisease"""
sql3 = """SELECT * FROM public.doctor_total_disease"""
sql4 = """SELECT * FROM medii.cris_dataset"""


curs.execute(sql1)
rows = curs.fetchall()
clinical_disease = pd.DataFrame(rows)
clinical_disease = clinical_disease.fillna("")


curs.execute(sql2)
rows = curs.fetchall()
diseasecode_disease = pd.DataFrame(rows)


curs.execute(sql3)
rows = curs.fetchall()
doctor_totaldisease = pd.DataFrame(rows)
doctor_totaldisease = doctor_totaldisease.fillna("")


curs.execute(sql4)
rows = curs.fetchall()
doctor_clinical = pd.DataFrame(rows)
doctor_clinical = doctor_clinical.fillna("")

#질병 코드와 한글 질병명을 매칭해주는 함수
def disease_match(text):
  text = text.split(', ')
  result = dict()

  for word in text:          
    disease_indexs = diseasecode_disease[diseasecode_disease['disease_code'] == word].index
    if(len(disease_indexs)):
      result[word] = diseasecode_disease['disease_kor'][disease_indexs[0]]

  return result

#의료진이 쓴 논문에 대해 점수를 매기는 함수

def paper_score(input, w1, w2):
  
  doctor_paper_data = doctor_totaldisease.copy()

  def preprocess(text):
    text = text.replace('.', "dot")

    return text

  def overlap_paper(text):
    paper_overlap = 0
    papers = text.split('/ ')
    for paper in papers:
      paper = paper.split(', ')
      if all( temp in paper for temp in std):
        paper_overlap += 1
    
    return paper_overlap


  def overlap_keyword(text):
    words_count = {}

    text = text.replace('/ ', ', ')
    words = text.split(', ')
    word_target = set(words)
    add_keyword = set(std) & word_target

    
    for word in words:
      if word in words_count:
        words_count[word] += 1
      else:
        words_count[word] = 1

    sorted_words = sorted([(k, v) for k, v in words_count.items()], key=lambda word_count: -word_count[1])
    keyword = [w for w in sorted_words if w[0] in add_keyword]
    if(len(keyword) >= 5):
      keyword = keyword[0:5]

    return keyword

  target_input = preprocess(input)
  target_name = list(doctor_paper_data['name_kor'])
  target_index = len(target_name)
  target_name.append('target')

  text = list(doctor_paper_data['paper_disease_all'])
  target_text = [ preprocess(t) for t in text ]
  target_text.append(target_input)

  doctors = pd.DataFrame({'name': target_name,
                          'text': target_text})

  tfidf_vector = TfidfVectorizer(min_df =3, max_features = 6000)
  tfidf_matrix = tfidf_vector.fit_transform(doctors['text']).toarray()

  cosine_sim = cosine_similarity(tfidf_matrix)
  cosine_sim_df = pd.DataFrame(cosine_sim, columns = doctors.name)
  cosine_sim_df.head()

  temp = cosine_sim_df['target'][0:target_index]
  print(temp)
  doctor_paper_data['cosine_simil'] = temp

  std = input.split(', ')


  doctor_paper_data['keyword_paper'] = doctor_paper_data.apply(lambda x: overlap_keyword(x['paper_disease_all']), axis = 1)
  doctor_paper_data['overlap_paper'] = doctor_paper_data.apply(lambda x: overlap_paper(x['paper_disease_all']), axis = 1)
  doctor_paper_data['total_paper'] =  doctor_paper_data.apply(lambda x: (x['paper_impact']*w1+x['cosine_simil']*w2)/(w1+w2), axis = 1)

  return doctor_paper_data


#의료진이 실시한 임상시험에 대해 점수를 매기는 함수

def clinical_score(title, input, option):
  
  doctor_disease_data =  pd.DataFrame({'chief_name': doctor_totaldisease['name_kor'],
                                       'belong': doctor_totaldisease['belong'],
                                      'clinical_count': doctor_totaldisease['clinical_count'],
                                      'clinical_disease_all': doctor_totaldisease['clinical_disease_all']
                                       })

  doctor_trial_data = doctor_clinical.copy()

  overlap_docter = list()

  if option == '1':
    doctor_indexs = doctor_trial_data[doctor_trial_data['brief_title_eng'] == title].index

    if len(doctor_indexs):
      for i in range(0, len(doctor_indexs)):
        overlap_docter.append([doctor_trial_data['chief_name_kor'][doctor_indexs[i]],doctor_trial_data['chief_belong_kor'][doctor_indexs[i]] ] )
      print("임상시험을 진행한 사람 : ", end = ' ')
      print(overlap_docter )
    else:
      print("임상시험 수행한 사람 검색 불가")

  def preprocess(text):

      text = text.replace('.', "dot")
      return text

  def remove(name, target, remv,  overlap_docter):
    if name in  overlap_docter:
      target = target.split('/ ')
      target.remove(remv)  

      return '/ '.join(target)
    
    return target

    
  target_input = preprocess(input)
  target_name = list(doctor_disease_data['chief_name'])
  target_index = len(target_name)
  target_name.append('target')

  if option == '1':
    doctor_disease_data['clinical_disease_all'] = doctor_disease_data.apply(lambda x: remove(x['chief_name'],x['clinical_disease_all'],input,  overlap_docter ), axis = 1)

  text = list(doctor_disease_data['clinical_disease_all'])
  target_text = [ preprocess(t) for t in text ]
  target_text.append(target_input)

  doctors = pd.DataFrame({'name': target_name,
                          'text': target_text})

  tfidf_vector = TfidfVectorizer(min_df =3, max_features = 6000)
  tfidf_matrix = tfidf_vector.fit_transform(doctors['text']).toarray()

  cosine_sim = cosine_similarity(tfidf_matrix)
  cosine_sim_df = pd.DataFrame(cosine_sim, columns = doctors.name)
  cosine_sim_df.head()

  doctor_disease_data['total_clinical'] = cosine_sim_df['target'][:target_index]


  std = input.split(', ')

  def overlap_clinical(text):
    clinical_overlap = 0
    clinicals = text.split('/ ')
    for clinical in clinicals:
      clinical = clinical.split(', ')
      if all( temp in clinical for temp in std):
        clinical_overlap += 1
    
    return clinical_overlap

  def overlap_keyword(text):
    text = text.replace('/ ', ', ')
    words = text.split(', ')
    word_target = set(words)
    add_keyword = set(std) & word_target

    words_count = {}
    for word in words:
      if word in words_count:
        words_count[word] += 1
      else:
        words_count[word] = 1

    sorted_words = sorted([(k, v) for k, v in words_count.items()], key=lambda word_count: -word_count[1])
    keyword = [w for w in sorted_words if w[0] in add_keyword]
    if(len(keyword) >= 5):
      keyword = keyword[0:5]

    return keyword
  
  doctor_disease_data['keyword_clinical'] = doctor_disease_data.apply(lambda x: overlap_keyword(x['clinical_disease_all']), axis = 1)
  doctor_disease_data['overlap_clinical'] = doctor_disease_data.apply(lambda x: overlap_clinical(x['clinical_disease_all']), axis = 1)
  doctor_disease_data['index'] = range(target_index)

  result = doctor_disease_data[['total_clinical', 'overlap_clinical']]

  return result

#추천을 위한 코드

def get_recommendation(input, option, weight_paper, weight_trial, weight_paper_impact, weight_sim):
  
  clinical_disease_data = clinical_disease.copy()

  trial_title = 'none'
  if(option == '1'):
    trial_indexs = clinical_disease_data[clinical_disease_data['title_eng'] == input].index

    if not(len(trial_indexs)):
      print('존재하지 않는 임상시험입니다')
      return -1

    trial_index = trial_indexs[0]
    trial_title = input
    input = clinical_disease_data['first_code_all'][trial_index]
    print(input)

    print('검색 제목 : ' + clinical_disease_data['title_kor'][trial_index])
  
  print('추출된 질병 : ', end = ' ')
  print('추출된 질병 (한글명 매칭): ', end = ' ')
  print(disease_match(input))

  paper_grade = paper_score(input, weight_paper_impact, weight_sim)
  clinical_grade = clinical_score(trial_title, input, option)
  
  person_grade = pd.concat([paper_grade,clinical_grade], axis = 1)
  person_grade['total_score'] = person_grade.apply( lambda x: (x['total_paper']*weight_paper + x['total_clinical']*weight_trial)/(weight_paper+weight_trial) , axis= 1) 
  ranking = person_grade.sort_values(by='total_score', ascending=False)
  return ranking[0:5]



input_text = ' '

while(input_text != 'exit'):
    print('\n\n옵션을 입력해주세요')
    print('1.임상시험명으로 추천받기    2.질병명 입력하여 추천받기')
    option = input()


    print('\n임상시험명 또는 검색하고자 하는 질병명들을 입력해주세요')
    input_text = input()

    print('\n가중치 비율을 입력하세요 ( 논문 점수 : 임상시험 점수 )')
    weight_paper = int(input('논문 점수 가중치 : '))
    weight_trial = int(input('임상시험 점수 가중치 : '))
    print('논문 점수 : 임상시험 점수 = ' + str(weight_paper) + ' : ' + str(weight_trial))

    print('\n가중치 비율을 입력하세요 ( 논문 impact : 질병 유사도 )')
    weight_paper_impact = int(input('논문 impact 가중치 : '))
    weight_sim = int(input('질병 유사도 가중치 : '))
    print('논문 impact : 질병 유사도 가중치 = ' + str(weight_paper_impact) + ' : ' + str(weight_sim))

    count = 1

    recom_list = get_recommendation(input_text,option, weight_paper, weight_trial, weight_paper_impact, weight_sim)

    print("---------------------------------------")

    for index in range(0,5):
      i = recom_list.iloc[index]
      print(str(count) + '순위')
      for key, value in i.items():
        print(key +' : '+ str(value))

      print('---------------------------------------')

      count += 1

    print('---------------------------------------')

