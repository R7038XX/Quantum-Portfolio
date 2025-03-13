###Streamlit
import streamlit as st
###基本的なライブラリのインポート
import numpy as np
import datetime
from datetime import datetime, timedelta, timezone
import heapq
import copy
#証券データの読み込みに利用するライブラリ
import pandas as pd
import pandas_datareader.data as web
### Amplify
# 決定変数の作成
from amplify import VariableGenerator
from amplify import one_hot, sum
from amplify import Model, FixstarsClient, solve
from amplify import equal_to, one_hot
###警告の非表示
import warnings
warnings.simplefilter('ignore')
###pyomo, ipopt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# FixstarsAmplify AE トークンの入手先
# https://amplify.fixstars.com/ja/
# https://amplify.fixstars.com/ja/register
# FixstarsAmplify AE トークン
# 実行マシンクライアントの設定
client = FixstarsClient()
client.token = AMPLIFY_TOKEN
# タイムアウト1秒
client.parameters.timeout = timedelta(seconds=10)

st.title("FinTech : Financial Portfolio")
st.write("# 世帯情報")
st.write("## 家族構成")

number_adult = int(st.number_input("大人[人]", min_value=0, max_value=10, step=1))
number_university = int(st.number_input("大学生[人]", min_value=0, max_value=5, step=1))
number_highschool = int(st.number_input("高校生[人]", min_value=0, max_value=5, step=1))
number_juniorhighschool = int(st.number_input("中学生[人]", min_value=0, max_value=5, step=1))
number_juniorschool = int(st.number_input("小学生[人]", min_value=0, max_value=5, step=1))
number_child = int(st.number_input("小児[人]", min_value=0, max_value=5, step=1))

st.write("## 世帯年収")
house_income = 0
house_income = int(st.number_input("世帯年収[円]", min_value=0, max_value=100000000, step=1))

#budget = int(st.number_input("予算入力してください。単位は円です。（最大:1千万円）", min_value=1, max_value=10000000, step=1))
budget  = 0
if house_income >0:
    budget  = 1000000
    st.write("世帯年収から計算で求めた予算は", budget, "円です")

is_high_risk_high_return = st.radio("", ("ハイリスク・ハイリターン", "ローリスク・ローリターン"), horizontal=True, args=[1, 0])

high_risk_preference = False
if is_high_risk_high_return =="ハイリスク・ハイリターン":
    high_risk_preference = True
    st.write("選択：ハイリスク・ハイリターン")
else:
    high_risk_preference = False
    st.write("選択：ローリスク・ローリターン")
st.write("")
st.write("")

st.write("# 算定期間")
calculate_period = st.slider("スライダーを動かして、算定期間（年）を選択", min_value=1, max_value=10)
st.write("選択された期間:", calculate_period,"年")

st.write("# 現状の資産")
columns = ["コード","保有数","銘柄名"]
edited_df = pd.DataFrame(columns=columns)
edited_df = st.data_editor(edited_df, num_rows="dynamic")

st.write("# マーケット")
st.multiselect('マーケットを選んでください',
            ['米国', '日本', '香港', '中国', 'シンガポール','豪州','カナダ','マレーシア'])

st.write("# 市場")
st.write("## 米国")
st.multiselect('米国の市場を選んでください',
            ['ナスダック', 'NYSE', 'AMEX', 'OTC', 'ETF'])  

st.write("## 日本")
st.multiselect('日本の市場を選んでください',
            ['ETF', '投資信託', '先物', '東証：プライム', '東証：スタンダード','東証：グロース','名証：プレミア','名証：メイン','名証：ネクスト','福証：本則市場','福証：Q-Board','福証：Fukuoka PRO Market','札証：本則市場','札証：アンビシャス'])  

st.write("# 業種")
st.multiselect('業種を選んでください',
            ['ALL', 'アパレル', 'アパレル小売', 'アルミ', '医療機器','医療機器・製品','医療小売','医療サービス','医療施設','医療流通','飲食店','REIT：ホテル','REIT:モーゲージ'])  


# 計算開始
### 予算金額に収まる銘柄一覧の作成
master_file_path = "./data/master/tokyo_prime_master.csv"
df_master = pd.read_csv(master_file_path)
#データの型変換
df_master['price_per_share']=df_master['price_per_share'].astype('int')
df_master['profit_ratio']=df_master['profit_ratio'].astype('float')
df_master['sharp_ratio']=df_master['sharp_ratio'].astype('float')
df_master['rsi']=df_master['rsi'].astype('float')
df_master['bband_ratio']=df_master['bband_ratio'].astype('float')
#print(df_master.dtypes)
# 予算金額以内のデータに絞り込み
query_str = ' price_per_share <=' + str(budget)
#st.write(query_str)
df_master=df_master.query(query_str)
### 予算金額以内の企業から、最終順位の昇順で上位100社選出。
#   最終順位:「利益率降順 かつ RSI昇順」の上位で、かつ、５年分データが揃っているもの。
df_master = df_master.head(100)
df_master = df_master.sort_values(by='final_order',ascending=True)
#st.dataframe(df_master)

st.write("予算で購入可能な企業一覧(上位100社)")
_df = df_master
_df = _df.sort_values(by='price_per_share',ascending=False)
#_df = _df.rename(columns={'price_per_share': '１株あたり価格（円）'})
_df = _df.iloc[:,0:10]
st.dataframe(_df)

run_flag = False
if "count" not in st.session_state: # (C)
    st.session_state.count = 0 # (A)
run_apps = st.button("計算")

if run_apps:
    ###　銘柄数
    N= df_master.shape[0]
    #st.write("行数: ", N)
    ###　銘柄コードを格納するリスト
    stockcodes=[]
    ###　銘柄コード、銘柄名を格納する辞書型データ
    code_name_dict = {}
    ### (終値ー前日終値)/前日 の５年分データ(return_rate)明細が、各社別に格納される。共分散計算に使われる。
    rates = []
    ### (終値ー前日終値)/前日 の５年分データ(return_rate)平均値が、各社別に格納される。
    profit_ratio=np.zeros(N)
    ### ５年分データに関するシャープレシオ平均値が、各社別に格納される。
    sharp_ratio=np.zeros(N)
    ### ５年分データに関するRSI/100の平均値が、各社別に格納される。
    rsi=np.zeros(N)
    ### ５年分データに関するボリンジャーバンド比率の平均値が、各社別に格納される。
    ### ボリンジャーバンド比率：3σに関する、 (upper - lower)/ upper の値
    bb_ratio=np.zeros(N)

    line_number = 0
    
    ###
    for index, row in df_master.iterrows():
        #銘柄情報格納
        sc = row['コード']
        brand_name = row['銘柄名']
        stockcodes.append(sc)
        code_name_dict[sc]=brand_name
        # 銘柄別のCSVファイルを読み込む設定
        # 共有ファイルのファイルIDを指定
        _file_path = row['file_path']
        # ファイルを取得し、ローカルに保存
        df_each = pd.read_csv(_file_path)
        df_each = df_each.sort_values(by='Date',ascending=True)

        #前日の終値と今日の終値を比較して、前日比を調べています。
        return_rate = np.zeros(len(df_each.values))
        for k in range(len(df_each.values)-1):
            return_rate[k+1] = (float(df_each.values[k+1][3])-float(df_each.values[k][3]))/float(df_each.values[k][3])
        rates.append(return_rate)
        ###ポートフォリオレシオ
        profit_ratio[line_number]=float(row['profit_ratio'])
        ### シャープレシオ
        sharp_ratio[line_number]=float(row['sharp_ratio'])
        ### RSI/100
        rsi[line_number]=float(row['rsi'])
        ### ボリンジャーバンドの3σ比率 (upper-lower)/upper
        bb_ratio[line_number]=float(row['bband_ratio'])
        
        if line_number > 0 and line_number % 5 ==0:
            print("データファイル:", line_number)
    
        line_number = line_number + 1 
    #print(stockcodes)
    #print(code_name_dict)
    ######### 量子アニーリングの処理 #########
    gen = VariableGenerator()
    q = gen.array("Binary", N)
    parameters = [1.0, 1.0, 1.0, 1.0]
    Q = [0, 0, 0, 0]
    ### sign 
    if high_risk_preference ==True:
        sign = -1
    else:
        sign = +1

    ###  共分散の値を最小化
    for i in range(N):
        for j in range(N):
            sum1 = 0
            for day in range(len(rates)):
                sum1 += (rates[i][day] - profit_ratio[i])*(rates[j][day] - profit_ratio[j])
            Q[0] += sign*q[i]*q[j]*sum1/len(rates[i])
    ###  ボリンジャーバンド
    for i in range(N):
        Q[1] = Q[1] + bb_ratio[i]*q[i]
    ###  シャープレシオ
    for i in range(N):
        Q[2] = Q[2] - sharp_ratio[i]*q[i]
    ### 制約条件（銘柄数N個のうち、半分の銘柄を選択するという制約条件、罰金法）
    sum2 = 0
    for i in range(N):
        sum2 += q[i]

    Q[3] =  equal_to(sum2,10)

    if high_risk_preference ==True:
        objective = Q[0]+parameters[2]*Q[2]
    else:
        objective = Q[0]+parameters[1]*Q[1]

    # モデル化
    # 目的関数 Objective, 制約条件 Constraints の組合せ最適化モデル
    model = Model(objective, parameters[3]*Q[3])
    # アニーリングマシンの実行
    result = solve(model, client)  # 問題を入力してマシンを実行

    if len(result) > 0:
        # 量子アニーリングが返した解を q に代入
        q_solutions = q.evaluate(result.best.values)
        q_solutions = np.array(q_solutions)
        #print(q_solutions)

    index = 0
    qa_answer_list=[]
    for value in q_solutions:
        if value == 1:
            qa_answer_list.append(code_name_dict[stockcodes[index]])
        index += 1

    #df_answer = pd.DataFrame({'answer':answer})

    #print(answer)
    #st.dataframe(df_answer)

    ### 10銘柄をよしなに振り分けるプログラム
    low_risk_brand_prices=[1111,1380,692,1918,669,1004,1189,322,2740,747]
    #ハイリスク・ハイリターンのTOP10銘柄の1株あたり価格
    high_risk_brand_prices=[4335,4760,1666,1870,2961,1258,1659,2233,419,2732]

    #予算100万円で仮置き
    budget=1000000

    model = pyo.ConcreteModel(name="NLP_problem", doc="10 variables")
    model.x = pyo.Var([0,1,2,3,4,5,6,7,8,9], domain=pyo.PositiveIntegers)

    summary1 = 0

    if high_risk_preference == True:
        for index in range(10):
            summary1 += high_risk_brand_prices[index] * model.x[index]
    else:
        for index in range(10):
            summary1 += low_risk_brand_prices[index] * model.x[index]
    
    average = summary1/10

    diff_sum = 0
    if high_risk_preference == True:
        for index in range(10):
            diff_sum +=  (average - high_risk_brand_prices[index]*model.x[index])**2
    else:
        for index in range(10):
            diff_sum +=  (average - low_risk_brand_prices[index]*model.x[index])**2

    lambda1 = 16.0
    lambda2 = 1.0

    model.OBJ = pyo.Objective(expr = lambda1 * (budget - summary1) + lambda2 * diff_sum, sense = pyo.minimize)
    model.Constraint = pyo.Constraint(expr = summary1 <= budget)

    # 最適化ソルバを設定
    opt = pyo.SolverFactory('ipopt')
    res = opt.solve(model) # 最適化計算を実行

    summary2 =0
    print("購入口数, １株あたり価格（円）、購入口数 x １株あたり価格")
    
    answer_list=[]
    if high_risk_preference == True:
        
        for index in range(10):
            
            print( int( model.x[index]() )  )
            summary2 += int(high_risk_brand_prices[index]) * int(model.x[index]())

            each_answer_list=[]
            each_answer_list.append(qa_answer_list[index])
            each_answer_list.append(high_risk_brand_prices[index])
            each_answer_list.append(int(model.x[index]()))
            each_answer_list.append(int(high_risk_brand_prices[index]) * int(model.x[index]()))
            answer_list.append(each_answer_list)
    else:
        for index in range(10):
            
            print( int( model.x[index]() )  )
            summary2 += int(low_risk_brand_prices[index]) * int(model.x[index]())

            each_answer_list=[]
            each_answer_list.append(qa_answer_list[index])
            each_answer_list.append(low_risk_brand_prices[index])
            each_answer_list.append(int(model.x[index]()))
            each_answer_list.append(int(low_risk_brand_prices[index]) * int(model.x[index]()))
            answer_list.append(each_answer_list)

    print("------------------")

    df_answer = pd.DataFrame(zip(* answer_list)).T
    df_answer.columns = ['銘柄名','１株あたり価格（円）','購入口数','小計（円）']
    st.dataframe(df_answer)
    st.write("購入金額合計（円）:" + str(summary2))

    print(answer_list)
    #print("summary_type,", type(summary2))
    print("購入金額合計（円）:", summary2)
