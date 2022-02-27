import pymysql
pymysql.install_as_MySQLdb()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', 50)


## 데이터베이스 연결
# doctor_total_disease 테이블 가져오기(의료진 정보 테이블)
connect = pymysql.connect(host='(주소)', user='(아이디)',
                          password='(비밀번호)', db='medii', charset='utf8mb4')
cursor = connect.cursor()
dataset = pd.read_sql_query('select * from doctor_total_disease', connect)

# totaldisease 테이블 가져오기(질병코드-질병정보 매칭)
conn = pymysql.connect(host='(주소)', user='(아이디)',
                          password='(비밀번호)', db='medii', charset='utf8mb4')
curs = conn.cursor(pymysql.cursors.DictCursor)
sql = """SELECT * FROM TotalDisease"""
curs.execute(sql)
rows = curs.fetchall()
disease_table = pd.DataFrame(rows)

# 널값제거(TF-IDF 구할 행렬들)
dataset['paper_disease_all'] = dataset['paper_disease_all'].fillna(' ')
dataset['clinical_disease_all'] = dataset['clinical_disease_all'].fillna(' ')
dataset['belong'] = dataset['belong'].fillna(' ')
dataset['major'] = dataset['major'].fillna(' ')


## 1.질병코드-질병명칭 매칭
def disease_match(input):
    input = input.split(', ')
    result = dict()
    for word in input:
        disease_indexs = disease_table[disease_table['disease_code'] == word].index
        if (len(disease_indexs)):
            result[word] = disease_table['disease_kor'][disease_indexs[0]]
    return result


## 2.논문과 유사도 계산
def paper_score(input):
    doctor_paper_data = dataset.copy()

    # 질병코드 토큰화 전처리
    def preprocess(text):
        text = text.replace('.', "dot")
        return text

    # 검색한 질병코드와 겹치는 논문 수 count
    def overlap_paper(text):
        paper_overlap = 0
        papers = text.split('/ ')
        for paper in papers:
            paper = paper.split(', ')
            if all(temp in paper for temp in std):
                paper_overlap += 1
        return paper_overlap

    # 검색한 질병코드 중, 의료진의 논문 대상질병코드와 겹치는 코드와 count 추출
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
        if (len(keyword) >= 5):
            keyword = keyword[0:5]
        return keyword

    # 질병코드와 KCD에 있는 한국어 질병명칭 매칭
    def disease_kor_match(text):
        for idx in range(0, len(text)):
            name_kor = disease_match(text[idx][0])
            if (len(name_kor) != 0):
                name_kor = name_kor[text[idx][0]]
            else:
                name_kor = 'x'
            text[idx] = [text[idx][0], name_kor, text[idx][1]]
        return text

    # 검색한 질병코드의 중분류와 유사도 계산
    def simil_large(text):
        target_input = text
        target_name = list(doctor_paper_data['name_kor'])
        target_index = len(target_name)
        target_name.append('target')

        text = list(doctor_paper_data['paper_disease_all'])
        target_text = [preprocess(t) for t in text]
        target_text.append(target_input)

        doctors = pd.DataFrame({'name': target_name,
                                'text': target_text})

        tfidf_vector = TfidfVectorizer(min_df=3, max_features=6000)
        tfidf_matrix = tfidf_vector.fit_transform(doctors['text']).toarray()

        cosine_sim = cosine_similarity(tfidf_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim, columns=doctors.name)

        temp = cosine_sim_df['target'][0:target_index]
        doctor_paper_data['cosine_simil_paper_large'] = temp

        return doctor_paper_data['cosine_simil_paper_large']

    simil_large(input)

    target_input = preprocess(input)
    target_name = list(doctor_paper_data['name_kor'])
    target_index = len(target_name)
    target_name.append('target')

    text = list(doctor_paper_data['paper_disease_all'])
    target_text = [preprocess(t) for t in text]
    target_text.append(target_input)

    doctors = pd.DataFrame({'name': target_name,
                            'text': target_text})

    tfidf_vector = TfidfVectorizer(min_df=3, max_features=6000)
    tfidf_matrix = tfidf_vector.fit_transform(doctors['text']).toarray()

    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, columns=doctors.name)
    cosine_sim_df.head()

    temp = cosine_sim_df['target'][0:target_index]
    doctor_paper_data['cosine_simil_paper'] = temp

    std = input.split(', ')

    doctor_paper_data['keyword_paper'] = doctor_paper_data.apply(lambda x: overlap_keyword(x['paper_disease_all']),
                                                                 axis=1)
    doctor_paper_data['keyword_paper'] = doctor_paper_data.apply(lambda x: disease_kor_match(x['keyword_paper']),
                                                                 axis=1)
    doctor_paper_data['overlap_paper'] = doctor_paper_data.apply(lambda x: overlap_paper(x['paper_disease_all']),
                                                                 axis=1)
    doctor_paper_data['total_paper'] = doctor_paper_data.apply(
        lambda x: (x['cosine_simil_paper'] * 0.7 + x['cosine_simil_paper_large'] * 0.3), axis=1)

    return doctor_paper_data


## 3.임상과 유사도 계산
def clinical_score(input):
    doctor_disease_data = pd.DataFrame({'chief_name': dataset['name_kor'],
                                        'belong': dataset['belong'],
                                        'clinical_count': dataset['clinical_count'],
                                        'clinical_disease_all': dataset['clinical_disease_all']
                                        })

    doctor_trial_data = dataset.copy()

    # 질병코드 토큰화 전처리
    def preprocess(text):
        text = text.replace('.', "dot")
        return text

    # 검색한 질병코드와 겹치는 임상시험 수 count
    def overlap_clinical(text):
        clinical_overlap = 0
        clinicals = text.split('/ ')
        for clinical in clinicals:
            clinical = clinical.split(', ')
            if all(temp in clinical for temp in std):
                clinical_overlap += 1
        return clinical_overlap

    # 검색한 질병코드 중, 의료진의 임상시험 대상질병코드와 겹치는 코드와 count 추출
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
        if (len(keyword) >= 5):
            keyword = keyword[0:5]
        return keyword

    # 질병코드와 KCD에 있는 한국어 질병명칭 매칭
    def disease_kor_match(text):
        for idx in range(0, len(text)):
            name_kor = disease_match(text[idx][0])
            if (len(name_kor) != 0):
                name_kor = name_kor[text[idx][0]]
            else:
                name_kor = 'x'
            text[idx] = [text[idx][0], name_kor, text[idx][1]]
        return text

    # 검색한 질병코드의 중분류와 유사도 계산
    def simil_large(text):
        target_input = text
        target_name = list(doctor_disease_data['chief_name'])
        target_index = len(target_name)
        target_name.append('target')

        text = list(doctor_disease_data['clinical_disease_all'])
        target_text = [preprocess(t) for t in text]
        target_text.append(target_input)

        doctors = pd.DataFrame({'name': target_name,
                                'text': target_text})

        tfidf_vector = TfidfVectorizer(min_df=3, max_features=6000)
        tfidf_matrix = tfidf_vector.fit_transform(doctors['text']).toarray()

        cosine_sim = cosine_similarity(tfidf_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim, columns=doctors.name)
        cosine_sim_df.head()

        doctor_disease_data['cosine_simil_clinical_large'] = cosine_sim_df['target'][:target_index]

        return doctor_disease_data['cosine_simil_clinical_large']

    simil_large(input)

    target_input = preprocess(input)
    target_name = list(doctor_disease_data['chief_name'])
    target_index = len(target_name)
    target_name.append('target')

    text = list(doctor_disease_data['clinical_disease_all'])
    target_text = [preprocess(t) for t in text]
    target_text.append(target_input)

    doctors = pd.DataFrame({'name': target_name, 'text': target_text})

    tfidf_vector = TfidfVectorizer(min_df=3, max_features=6000)
    tfidf_matrix = tfidf_vector.fit_transform(doctors['text']).toarray()

    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, columns=doctors.name)
    cosine_sim_df.head()

    doctor_disease_data['cosine_simil_clinical'] = cosine_sim_df['target'][:target_index]

    std = input.split(', ')

    doctor_disease_data['keyword_clinical'] = doctor_disease_data.apply(
        lambda x: overlap_keyword(x['clinical_disease_all']), axis=1)
    doctor_disease_data['keyword_clinical'] = doctor_disease_data.apply(
        lambda x: disease_kor_match(x['keyword_clinical']), axis=1)
    doctor_disease_data['overlap_clinical'] = doctor_disease_data.apply(
        lambda x: overlap_clinical(x['clinical_disease_all']), axis=1)
    doctor_disease_data['total_clinical'] = doctor_disease_data.apply(
        lambda x: (x['cosine_simil_clinical'] * 0.7 + x['cosine_simil_clinical_large'] * 0.3), axis=1)
    doctor_disease_data['index'] = range(target_index)
    result = doctor_disease_data[['total_clinical', 'overlap_clinical', 'keyword_clinical']]

    return result


## 추천코드
def get_recommendation(input, weight_paper, weight_trial):
    print('추출된 질병 (한글명 매칭): ', end=' ')
    print(disease_match(input))

    paper_grade = paper_score(input)
    clinical_grade = clinical_score(input)

    person_grade = pd.concat([paper_grade, clinical_grade], axis=1)
    person_grade['total_score'] = person_grade.apply(
        lambda x: (x['total_paper'] * weight_paper + x['total_clinical'] * weight_trial) / (
                    weight_paper + weight_trial), axis=1)
    ranking = person_grade.sort_values(by='total_score', ascending=False)
    return ranking[0:5]


## 실행 코드
input_text = ' '
while (input_text != 'exit'):
    input_text = input("\n질병코드를 (, )로 구분해서 입력하세요(심장관련 코드: I00-I99, Q20-Q28) > ")
    print('\n가중치 비율을 입력하세요 ( 논문 점수 : 임상시험 점수 )')
    weight_paper = int(input('논문 점수 가중치 : '))
    weight_trial = int(input('임상시험 점수 가중치 : '))
    print('논문 점수 : 임상시험 점수 = ' + str(weight_paper) + ' : ' + str(weight_trial))

    count = 1
    recom_list = get_recommendation(input_text, weight_paper, weight_trial)
    print("---------------------------------------")
    for index in range(0, 5):
        i = recom_list.iloc[index]
        print('> ' + str(count) + '순위')
        for key, value in i.items():
            print(key + ' : ' + str(value))
        print('---------------------------------------')

        count += 1