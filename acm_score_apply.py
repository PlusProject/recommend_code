import pandas as pd
import time
import pymysql
pymysql.install_as_MySQLdb()

## 데이터베이스 연결
# doctor_all2 테이블 가져오기(의료진 정보 테이블)
connect = pymysql.connect(host='database-skku.c6dzc5dnqf69.ap-northeast-2.rds.amazonaws.com', user='admin',
                          password='tjdrbsrhkseo123', db='medii', charset='utf8mb4')
cursor = connect.cursor()
df = pd.read_sql_query('select * from medii.doctor_all2', connect)

# totaldisease 테이블 가져오기(질병코드-질병정보 매칭)
conn = pymysql.connect(host='database-skku.c6dzc5dnqf69.ap-northeast-2.rds.amazonaws.com', user='admin',
                       password='tjdrbsrhkseo123', db='medii', charset='utf8mb4')
curs = conn.cursor(pymysql.cursors.DictCursor)
sql = """SELECT * FROM TotalDisease"""
curs.execute(sql)
rows = curs.fetchall()
disease_table = pd.DataFrame(rows)


print(df.columns)
input_text = ' '
time1 = time.time()




# 입력받은 질병코드들의 한글 질병명을 각각 찾아서 매칭
def disease_match(text):
    text = text.split(', ')
    result = dict()
    for word in text:
        disease_indexs = disease_table[disease_table['disease_code'] == word].index
        if(len(disease_indexs)):
            result[word] = disease_table['disease_kor'][disease_indexs[0]]
    return result
        
# 질병코드 하나의 한글 질병명을 각각 찾아서 매칭
def disease_match_one(text):
    disease_indexs = disease_table[disease_table['disease_code'] == text].index
    if(len(disease_indexs)):
        return " {" + disease_table['disease_kor'][disease_indexs[0]] + "} "
    else:
        return " {-} "
    
# 새로운 추천 알고리즘
def get_recommendation(input, weight_paper, weight_trial):
    # 불필요한 칼럼 삭제(입력 코드에 해당하지 않는 a~z 칼럼)
    big_codes = []
    for input_code in input:
        big_codes.append((input_code[0]).upper())
    # big_codes = 입력받은 질병 코드 중 대분류 코드만 추출하여 저장
    for i in range(ord('A'),ord('Z')+1):
        if chr(i) not in big_codes:
            df.drop(chr(i), inplace=True, axis=1)

    def overlap(text):
        if text=="":
            return 0
        count = 0
        dic = eval(text)
        input_code = "\'"+input+"\'"
        if input_code in dic:
            count = dic[input_code]
        return count

    def calcul_sim(x, y):
        x = x[2:]
        if(x == y):
            return 100
        if x.split('.')[0] == y.split('.')[0]:
            return 1
        if x[0] == y[0]:
            return 0.01
        return 0
    
    print('추출된 질병 (한글명 매칭): ', end=' ')
    print(disease_match(input))
    
    # df 초기화
    df['o_p'] = 0.0
    df['real_o_p'] = 0.0
    df['o_c'] = 0.0
    df['real_o_c'] = 0.0
    df['top1'] = ""
    df['top2'] = ""
    df['top3'] = ""
    df['explain1'] = ""
    df['explain2'] = ""
    df['explain3'] = ""
    df['explainp'] = ""
    df['explainc'] = ""
    df['paper_count'] = df['paper_count'].fillna(0)
    df['clinical_count'] = df['clinical_count'].fillna(0)
    df['paper_impact'] = df['paper_impact'].fillna(0.0)
    df['paper_disease_all'] = df['paper_disease_all'].fillna("")
    df['clinical_disease_all'] = df['clinical_disease_all'].fillna("")
    df['clinical_allcount'] = df['clinical_allcount'].fillna("")
    df['paper_allcount'] = df['paper_allcount'].fillna("")
    inputs = input.split(', ')

    code_num = len(input)
    for input_code in inputs:
        for i in df.index:
            # 대분류 도입
            input_big = input_code[0].upper()
            codes = eval(df[input_big][i])
            ptemp = 0.0
            ctemp = 0.0
            # 논문/임상시험 가중치 계산
            for code in codes: 
                sim = calcul_sim(code, input_code)
                temp = sim*codes[code]
                if code[0]=='p':
                    ptemp+=temp
                else:
                    ctemp+=temp
            df['o_c'][i] = ctemp
            df['o_p'][i] = ptemp
        
        if(code_num>1):
            mean_score = df['o_p'].mean()
            std_score = df['o_p'].std()
            df['o_p'] = (df['o_p']-mean_score)/std_score
            mean_score = df['o_c'].mean()
            std_score = df['o_c'].std()
            df['o_c'] = (df['o_c']-mean_score)/std_score
        df['real_o_p'] += df['o_p']
        df['real_o_c'] += df['o_c']
        
    wp = float(weight_paper)
    wt = float(weight_trial)
    wpt = wp+wt
    df['o_p'] = df['real_o_p']
    df['o_c'] = df['real_o_c']
    
    pmax = df['o_p'].max()
    pmin = df['o_p'].min()
    pweight = 100*wp/((pmax-pmin)*wpt)
    df['o_p'] = df['o_p']-pmin
    df['o_p'] = df['o_p'] * pweight
    
    cmax = df['o_c'].max()
    cmin = df['o_c'].min()
    cweight = 100*wt/((cmax-cmin)*wpt)
    df['o_c'] = df['o_c']-cmin
    df['o_c'] = df['o_c']*cweight
    df['total_score'] = df['o_p'] + df['o_c']
    
    sorted_df = df.sort_values(
        by=['total_score'], axis=0, ascending=False)[0:20]
    sorted_df = sorted_df.reset_index()
    total_total_score = sorted_df['total_score'].sum()
    
    # 정규화
    tmax = sorted_df['total_score'].max()
    tmin = sorted_df['total_score'].min()
    sorted_df['total_ratio'] = sorted_df['total_score']/total_total_score
    
    for i in sorted_df.index:
        dic = eval(sorted_df['disease'][i])
        dic_code = {}
        delete = []
        for j in dic:
            t=0.0
            for input_code in input:
                if j[0]=='p':
                    t += float(dic[j]) * float(calcul_sim(j, input_code)) * pweight
                else:
                    t += float(dic[j]) * float(calcul_sim(j, input_code)) * cweight
            code_only = j[2:]
            if t>0:
                if code_only in dic_code:
                    dic_code[code_only] = float(round(dic_code[code_only]+t, 2))
                else:
                    dic_code[code_only] = float(round(t, 2))
                    
                
            if(t == 0.0): delete.append(j)
        sdic = sorted(
            dic_code.items(), key=lambda x: x[1], reverse=True)[0:5]
        explain = ""
        for j in sdic:
            explain += disease_match_one(j[0])
        codes = ", ".join([str(_) for _ in sdic]).replace('p-','논문-').replace('t-','임상-')
        codes = codes.replace('\',',':').replace('(','').replace(')','').replace(',',' ')
        print(codes)
        if len(codes.split(" "))>=2:
            code1 = (codes.split(" ")[0] + " "+ codes.split(" ")[1]).lstrip("\'")
            explain1 = (explain.split("} ")[0])[2:]
        else:
            code1 = " "
            explain1 = " "
        if len(codes.split(" "))>=5:
            code2 = (codes.split(" ")[3] + " "+ codes.split(" ")[4]).lstrip("\'")
            explain2 = (explain.split("} ")[1])[2:]
        else:
            code2 = " "
            explain2 = " "
        if len(codes.split(" "))>=8:
            code3 = (codes.split(" ")[6] + " "+ codes.split(" ")[7]).lstrip("\'")
            explain3 = (explain.split("} ")[2])[2:]
        else:
            code3 = " "
            explain3 = " "
        if len(explain1)<5:
            explain1 = " "
        if len(explain2)<5:
            explain2 = " "
        if len(explain3)<5:
            explain3 = " "
        sorted_df['top1'][i] = code1 
        sorted_df['top2'][i] = code2
        sorted_df['top3'][i] = code3
        sorted_df['explain1'][i] = explain1
        sorted_df['explain2'][i] = explain2
        sorted_df['explain3'][i] = explain3


        sorted_df['explainp'][i] = overlap(sorted_df['paper_allcount'][i])
        sorted_df['explainc'][i] = overlap(sorted_df['clinical_allcount'][i])
        
        if (overlap(sorted_df['paper_allcount'][i]) > sorted_df['paper_count'][i]):
            sorted_df['explainp'][i] = "("+str(int(sorted_df['paper_count'][i])) + "건)"
        if (overlap(sorted_df['paper_allcount'][i]) <= sorted_df['paper_count'][i]):
            sorted_df['explainp'][i] = "("+str(overlap(sorted_df['paper_allcount'][i])) + "건)"

        if (overlap(sorted_df['clinical_allcount'][i]) > sorted_df['clinical_count'][i]):
            sorted_df['explainc'][i] = "("+str(int(sorted_df['clinical_count'][i])) + "건)"
        if (overlap(sorted_df['clinical_allcount'][i]) <= sorted_df['clinical_count'][i]):
            sorted_df['explainc'][i] = "("+str(overlap(sorted_df['clinical_allcount'][i])) + "건)"
        


        sorted_df['name_kor'][i] = sorted_df['name_kor'][i]
        sorted_df['major'][i] = codes
        sorted_df['o_p'][i] = str(round(sorted_df['o_p'][i],2)) 
        sorted_df['o_c'][i] = str(round(sorted_df['o_c'][i],2)) 
        sorted_df['total_score'][i] = str(round(sorted_df['total_score'][i],2))  
        sorted_df['paper_impact'][i] = str(round(float(sorted_df['paper_impact'][i]), 2))
    
    sorted_df['ranking'] = range(1, 21)           
    temp = sorted_df.to_json(orient='records')
    time5 = time.time()    
    print("총 소요 시간: " + str(round(time5-time1,3)) + "초")            
    return temp
## 실행코드

while (input_text != 'exit'):
    input_text = input("\n질병코드를 (, )로 구분해서 입력하세요(심장관련 코드: I00-I99, Q20-Q28) > ")
    print('\n가중치 비율을 입력하세요 ( 논문 점수 : 임상시험 점수 )')
    weight_paper = int(input('논문 점수 가중치 : '))
    weight_trial = int(input('임상시험 점수 가중치 : '))
    print('논문 점수 : 임상시험 점수 = ' + str(weight_paper) + ' : ' + str(weight_trial))
    print("---------------------------------------")
    get_recommendation(input_text,weight_paper,weight_trial)