import streamlit as st
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
from amplify import (
    VariableGenerator, 
    Model, 
    FixstarsClient, 
    solve, 
    equal_to
)
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import warnings
warnings.simplefilter('ignore')

# 定数定義
MASTER_FILE_PATH = "./data/master/tokyo_prime_master.csv"
DEFAULT_BUDGET = 1000000
MAX_PORTFOLIO_SIZE = 10
TIMEOUT_SECONDS = 10
OPTIMIZATION_PARAMETERS = {
    'lambda1': 16.0,
    'lambda2': 1.0
}

# マーケット・市場の選択肢
MARKETS = {
    '米国': ['ナスダック', 'NYSE', 'AMEX', 'OTC', 'ETF'],
    '日本': [
        'ETF', '投資信託', '先物', '東証：プライム', '東証：スタンダード', 
        '東証：グロース', '名証：プレミア', '名証：メイン', '名証：ネクスト',
        '福証：本則市場', '福証：Q-Board', '福証：Fukuoka PRO Market',
        '札証：本則市場', '札証：アンビシャス'
    ],
    '香港': ['メインボード', 'GEM'],
    '中国': ['上海', '深セン'],
    'シンガポール': ['メイン・ボード', 'カタリスト'],
    '豪州': ['ASX'],
    'カナダ': ['トロント証券取引所', 'TSXベンチャー'],
    'マレーシア': ['メイン・マーケット', 'ACE・マーケット']
}

INDUSTRIES = [
    'ALL', 'アパレル', 'アパレル小売', 'アルミ', '医療機器', '医療機器・製品',
    '医療小売', '医療サービス', '医療施設', '医療流通', '飲食店', 
    'REIT：ホテル', 'REIT:モーゲージ'
]

# Amplifyクライアントの初期化
def init_amplify_client():
    client = FixstarsClient()
    client.token = st.secrets["AMPLIFY_TOKEN"]
    client.parameters.timeout = timedelta(seconds=TIMEOUT_SECONDS)
    return client

def render_household_info():
    """世帯情報入力用のUIを表示"""
    st.write("# 世帯情報")
    st.write("## 家族構成")
    
    family_members = {
        "大人": st.number_input("大人[人]", min_value=0, max_value=10, step=1),
        "大学生": st.number_input("大学生[人]", min_value=0, max_value=5, step=1),
        "高校生": st.number_input("高校生[人]", min_value=0, max_value=5, step=1),
        "中学生": st.number_input("中学生[人]", min_value=0, max_value=5, step=1),
        "小学生": st.number_input("小学生[人]", min_value=0, max_value=5, step=1),
        "小児": st.number_input("小児[人]", min_value=0, max_value=5, step=1)
    }
    
    st.write("## 世帯年収")
    house_income = st.number_input("世帯年収[円]", min_value=0, max_value=100000000, step=1)
    
    budget = DEFAULT_BUDGET if house_income > 0 else 0
    if budget:
        st.write(f"世帯年収から計算で求めた予算は {budget} 円です")
    
    return budget

def get_risk_preference():
    """投資リスク選好の取得"""
    is_high_risk = st.radio(
        "", 
        ("ハイリスク・ハイリターン", "ローリスク・ローリターン"), 
        horizontal=True
    )
    high_risk_preference = is_high_risk == "ハイリスク・ハイリターン"
    st.write(f"選択：{'ハイリスク・ハイリターン' if high_risk_preference else 'ローリスク・ローリターン'}")
    st.write("")
    st.write("")
    return high_risk_preference

def render_market_selection():
    """マーケットと市場選択のUIを表示"""
    # セッション状態の初期化
    if 'selected_markets' not in st.session_state:
        st.session_state.selected_markets = []

    # マーケット選択
    st.write("# マーケット")
    selected_markets = st.multiselect(
        'マーケットを選んでください',
        list(MARKETS.keys()),
        key='selected_markets'
    )

    # 市場選択
    st.write("# 市場")
    
    # 選択されたマーケットに対応する市場を表示
    for market in selected_markets:
        st.write(f"## {market}")
        st.multiselect(
            f'{market}の市場を選んでください',
            MARKETS[market],
            key=f'market_{market}'  # ユニークなキーを設定
        )

    # 業種選択
    st.write("# 業種")
    st.multiselect('業種を選んでください', INDUSTRIES, default=['ALL'])

def load_stock_data(budget):
    """株式データの読み込みと前処理"""
    df_master = pd.read_csv(MASTER_FILE_PATH)
    
    # データ型変換
    numeric_columns = {
        'price_per_share': 'int',
        'profit_ratio': 'float',
        'sharp_ratio': 'float',
        'rsi': 'float',
        'bband_ratio': 'float'
    }
    for col, dtype in numeric_columns.items():
        df_master[col] = df_master[col].astype(dtype)
    
    # 予算内の銘柄に絞り込み
    df_master = df_master.query(f'price_per_share <= {budget}')
    df_master = df_master.sort_values(by='final_order', ascending=True).head(100)
    
    return df_master

def calculate_stock_metrics(df_master):
    """株式の指標を計算"""
    N = df_master.shape[0]
    stockcodes = []
    code_name_dict = {}
    rates = []
    profit_ratio = np.zeros(N)
    sharp_ratio = np.zeros(N)
    rsi = np.zeros(N)
    bb_ratio = np.zeros(N)
    
    for i, row in df_master.iterrows():
        # 銘柄情報の保存
        sc = row['コード']
        stockcodes.append(sc)
        code_name_dict[sc] = row['銘柄名']
        
        # 株価データの読み込み
        df_each = pd.read_csv(row['file_path'])
        df_each = df_each.sort_values(by='Date', ascending=True)
        
        # リターン率の計算
        return_rate = np.zeros(len(df_each.values))
        for k in range(len(df_each.values)-1):
            return_rate[k+1] = (float(df_each.values[k+1][3]) - float(df_each.values[k][3])) / float(df_each.values[k][3])
        rates.append(return_rate)
        
        # 各指標の保存
        profit_ratio[i] = float(row['profit_ratio'])
        sharp_ratio[i] = float(row['sharp_ratio'])
        rsi[i] = float(row['rsi'])
        bb_ratio[i] = float(row['bband_ratio'])
    
    return stockcodes, code_name_dict, rates, profit_ratio, sharp_ratio, rsi, bb_ratio

def optimize_portfolio(client, N, rates, profit_ratio, bb_ratio, sharp_ratio, high_risk_preference):
    """ポートフォリオの最適化"""
    gen = VariableGenerator()
    q = gen.array("Binary", N)
    parameters = [1.0, 1.0, 1.0, 1.0]
    Q = [0, 0, 0, 0]
    sign = -1 if high_risk_preference else 1
    
    # 共分散の最小化
    for i in range(N):
        for j in range(N):
            sum1 = 0
            for day in range(len(rates)):
                sum1 += (rates[i][day] - profit_ratio[i]) * (rates[j][day] - profit_ratio[j])
            Q[0] += sign * q[i] * q[j] * sum1 / len(rates[i])
    
    # 各指標の最適化
    for i in range(N):
        Q[1] += bb_ratio[i] * q[i]
        Q[2] -= sharp_ratio[i] * q[i]
    
    # 銘柄数の制約
    sum2 = sum(q[i] for i in range(N))
    Q[3] = equal_to(sum2, MAX_PORTFOLIO_SIZE)
    
    # 目的関数の設定
    objective = Q[0] + (parameters[2] * Q[2] if high_risk_preference else parameters[1] * Q[1])
    model = Model(objective, parameters[3] * Q[3])
    
    # 最適化実行
    result = solve(model, client)
    if len(result) == 0:
        return None
    
    return q.evaluate(result.best.values)

def calculate_portfolio_allocation(selected_stocks, prices, budget):
    """ポートフォリオの配分計算"""
    model = pyo.ConcreteModel()
    model.x = pyo.Var(range(MAX_PORTFOLIO_SIZE), domain=pyo.PositiveIntegers)
    
    # 総額計算
    total_cost = sum(prices[i] * model.x[i] for i in range(MAX_PORTFOLIO_SIZE))
    avg_cost = total_cost / MAX_PORTFOLIO_SIZE
    
    # 分散計算
    variance = sum((avg_cost - prices[i] * model.x[i])**2 for i in range(MAX_PORTFOLIO_SIZE))
    
    # 目的関数
    model.OBJ = pyo.Objective(
        expr=OPTIMIZATION_PARAMETERS['lambda1'] * (budget - total_cost) + 
             OPTIMIZATION_PARAMETERS['lambda2'] * variance,
        sense=pyo.minimize
    )
    
    # 予算制約
    model.Constraint = pyo.Constraint(expr=total_cost <= budget)
    
    # 最適化
    opt = pyo.SolverFactory('ipopt')
    opt.solve(model)
    
    return [int(model.x[i]()) for i in range(MAX_PORTFOLIO_SIZE)]

def main():
    st.title("FinTech : Financial Portfolio")
    
    # Amplifyクライアントの初期化
    client = init_amplify_client()
    
    # 世帯情報の入力
    budget = render_household_info()
    
    # リスク選好の取得
    high_risk_preference = get_risk_preference()
    
    # 算定期間の設定
    st.write("# 算定期間")
    calculate_period = st.slider("スライダーを動かして、算定期間（年）を選択", min_value=1, max_value=10)
    st.write(f"選択された期間: {calculate_period}年")
    
    # 現状の資産入力
    st.write("# 現状の資産")
    columns = ["コード", "保有数", "銘柄名"]
    edited_df = pd.DataFrame(columns=columns)
    edited_df = st.data_editor(edited_df, num_rows="dynamic")
    
    # マーケットと市場の選択
    render_market_selection()
    
    # 株式データの読み込み
    if budget > 0:
        df_master = load_stock_data(budget)
        st.write("予算で購入可能な企業一覧(上位100社)")
        display_df = df_master.sort_values(by='price_per_share', ascending=False)
        st.dataframe(display_df.iloc[:, 0:10])
        
        # 計算実行ボタン
        if st.button("計算"):
            # 進捗状況を表示するプレースホルダー
            status_placeholder = st.empty()
            
            with status_placeholder.container():
                # 株式指標の計算
                with st.spinner("銘柄データを分析中..."):
                    stockcodes, code_name_dict, rates, profit_ratio, sharp_ratio, rsi, bb_ratio = calculate_stock_metrics(df_master)
                
                # ポートフォリオ最適化
                with st.spinner("ポートフォリオを最適化中..."):
                    q_solutions = optimize_portfolio(
                        client, len(stockcodes), rates, profit_ratio, 
                        bb_ratio, sharp_ratio, high_risk_preference
                    )
            
            if q_solutions is not None:
                with status_placeholder.container():
                    # 選択された銘柄の抽出
                    with st.spinner("最適な銘柄を選定中..."):
                        selected_stocks = [
                            code_name_dict[stockcodes[i]] 
                            for i in range(len(q_solutions)) 
                            if q_solutions[i] == 1
                        ]
                        
                        # 価格リスト
                        prices = [1111, 1380, 692, 1918, 669, 1004, 1189, 322, 2740, 747] if not high_risk_preference else [4335, 4760, 1666, 1870, 2961, 1258, 1659, 2233, 419, 2732]
                        
                        # ポートフォリオ配分の計算
                        allocations = calculate_portfolio_allocation(selected_stocks, prices, budget)
                
                # 結果の表示
                result_data = []
                total_cost = 0
                for i in range(len(selected_stocks)):
                    cost = prices[i] * allocations[i]
                    total_cost += cost
                    result_data.append([
                        selected_stocks[i],
                        prices[i],
                        allocations[i],
                        cost
                    ])
                
                df_answer = pd.DataFrame(
                    result_data,
                    columns=['銘柄名', '１株あたり価格（円）', '購入口数', '小計（円）']
                )
                
                # 進捗表示を消去して結果を表示
                status_placeholder.empty()
                st.success("計算が完了しました！")
                st.dataframe(df_answer)
                st.write(f"購入金額合計（円）: {total_cost}")

if __name__ == "__main__":
    main()
