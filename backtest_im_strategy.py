import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from tqsdk import TqApi, TqAuth
import matplotlib.pyplot as plt
from fpdf import FPDF
import matplotlib

import datetime
import os

# 计算指标
def calculate_indicators(df):
    bb = BollingerBands(close=df['close'], window=26, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['rsi'] = RSIIndicator(close=df['close'], window=7).rsi()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ])
    df['avg_vol_12'] = df['volume'].rolling(window=12).mean()
    return df

# 回测主逻辑
def backtest(df, initial_equity=500000):
    df = calculate_indicators(df)
    position = 0
    equity = initial_equity
    shares = 0
    trades = []
    equity_curve = [equity]
    drawdown_curve = [0]
    peak = equity
    last_trade_day = None
    trade_days = set()
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        # 记录交易日
        trade_days.add(row.name.date())
        # 开仓
        if position == 0:
            # 多头
            if (prev['close'] > prev['bb_upper']) and (prev['rsi'] < 80) and (prev['volume'] > 1.1 * prev['avg_vol_12']) and (prev['tr'] < 80):
                entry = row['open']
                stop = entry - 2 * row['atr']
                position = 1
                shares = equity // entry
                equity -= shares * entry
                trades.append({'entry': entry, 'stop': stop, 'pos': 1, 'bar': i, 'date': row.name})
            # 空头
            elif (prev['close'] < prev['bb_lower']) and (prev['rsi'] > 20) and (prev['tr'] < 80):
                entry = row['open']
                stop = entry + 2 * row['atr']
                position = -1
                shares = equity // entry
                equity += shares * entry
                trades.append({'entry': entry, 'stop': stop, 'pos': -1, 'bar': i, 'date': row.name})
        # 持有多头
        elif position == 1:
            # 止损或最大亏损
            if (row['low'] <= trades[-1]['stop']) or ((row['close'] - trades[-1]['entry']) <= -80):
                exit_price = max(row['low'], trades[-1]['entry'] - 80)
                equity += shares * exit_price
                trades[-1]['exit'] = exit_price
                trades[-1]['exit_bar'] = i
                trades[-1]['profit'] = (exit_price - trades[-1]['entry']) * shares
                trades[-1]['exit_date'] = row.name
                position = 0
            else:
                # 跟踪止盈
                new_stop = row['close'] - 2 * row['atr']
                if new_stop > trades[-1]['stop']:
                    trades[-1]['stop'] = new_stop
        # 持有空头
        elif position == -1:
            if (row['high'] >= trades[-1]['stop']) or ((trades[-1]['entry'] - row['close']) <= -80):
                exit_price = min(row['high'], trades[-1]['entry'] + 80)
                equity -= shares * exit_price
                trades[-1]['exit'] = exit_price
                trades[-1]['exit_bar'] = i
                trades[-1]['profit'] = (trades[-1]['entry'] - exit_price) * shares
                trades[-1]['exit_date'] = row.name
                position = 0
            else:
                new_stop = row['close'] + 2 * row['atr']
                if new_stop < trades[-1]['stop']:
                    trades[-1]['stop'] = new_stop
        # 资金曲线
        equity_curve.append(equity)
        peak = max(peak, equity)
        drawdown_curve.append((peak - equity) / peak)
    # 平未平仓
    if position != 0:
        exit_price = df.iloc[-1]['close']
        if position == 1:
            equity += shares * exit_price
            trades[-1]['profit'] = (exit_price - trades[-1]['entry']) * shares
        else:
            equity -= shares * exit_price
            trades[-1]['profit'] = (trades[-1]['entry'] - exit_price) * shares
        trades[-1]['exit'] = exit_price
        trades[-1]['exit_bar'] = len(df) - 1
        trades[-1]['exit_date'] = df.index[-1]
    # 统计
    total_profit = sum(t.get('profit', 0) for t in trades)
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
    win_rate = winning_trades / total_trades if total_trades else 0
    profit_factor = sum(t['profit'] for t in trades if t['profit'] > 0) / abs(sum(t['profit'] for t in trades if t['profit'] < 0)) if any(t['profit'] < 0 for t in trades) else float('inf')
    max_drawdown = max(drawdown_curve)
    start_date = df.index[0].date()
    end_date = df.index[-1].date()
    days = (end_date - start_date).days + 1
    avg_trades_per_day = total_trades / days if days else 0
    return {
        'equity_curve': equity_curve,
        'drawdown_curve': drawdown_curve,
        'start_date': start_date,
        'end_date': end_date,
        'initial_equity': initial_equity,
        'final_equity': equity,
        'max_drawdown': max_drawdown,
        'total_profit': total_profit,
        'profit_rate': (equity - initial_equity) / initial_equity,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'trade_days': len(trade_days),
        'avg_trades_per_day': avg_trades_per_day,
        'trades': trades
    }

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 绘图
def plot_curves(equity_curve, drawdown_curve, filename_prefix):
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title('资金曲线')
    plt.legend()
    plt.tight_layout()
    eq_path = f"{filename_prefix}_equity.png"
    plt.savefig(eq_path)
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.plot(drawdown_curve, label='Drawdown Curve')
    plt.title('回撤曲线')
    plt.legend()
    plt.tight_layout()
    dd_path = f"{filename_prefix}_drawdown.png"
    plt.savefig(dd_path)
    plt.close()
    return eq_path, dd_path

# 生成英文PDF报告
def generate_pdf(report, eq_path, dd_path, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Backtest Report(SHFE.cu2201)", ln=1, align='C')
    pdf.cell(0, 10, f"Period: {report['start_date']} ~ {report['end_date']}", ln=1)
    pdf.cell(0, 10, f"Initial Equity: {report['initial_equity']}", ln=1)
    pdf.cell(0, 10, f"Max Drawdown: {report['max_drawdown']:.2%}", ln=1)
    pdf.cell(0, 10, f"Total Profit: {report['total_profit']:.2f}", ln=1)
    pdf.cell(0, 10, f"Profit Rate: {report['profit_rate']:.2%}", ln=1)
    pdf.cell(0, 10, f"Profit Factor: {report['profit_factor']:.2f}", ln=1)
    pdf.cell(0, 10, f"Trade Days: {report['trade_days']}", ln=1)
    pdf.cell(0, 10, f"Avg Trades/Day: {report['avg_trades_per_day']:.2f}", ln=1)
    pdf.cell(0, 10, f"Total Trades: {report['total_trades']}", ln=1)
    pdf.cell(0, 10, f"Winning Trades: {report['winning_trades']}", ln=1)
    pdf.cell(0, 10, f"Win Rate: {report['win_rate']:.2%}", ln=1)
    pdf.image(eq_path, x=10, y=pdf.get_y()+5, w=180)
    pdf.ln(60)
    pdf.image(dd_path, x=10, y=pdf.get_y()+5, w=180)
    pdf.output(pdf_path)

# import os
# from fpdf import FPDF

# 生成中文PDF报告
# 确保 simhei.ttf 字体文件在同目录下
# def generate_pdf(report, eq_path, dd_path, pdf_path):
#     pdf = FPDF()
#     pdf.add_page()
#     # 注册并使用中文字体
#     font_path = os.path.join(os.path.dirname(__file__), "simhei.ttf")  # 确保 simhei.ttf 在同目录
#     pdf.add_font("SimHei", "", font_path, uni=True)
#     pdf.set_font("SimHei", size=12)
#     pdf.cell(0, 10, "量化回测报告(SHFE.cu2201)", ln=1, align='C')
#     pdf.cell(0, 10, f"回测区间: {report['start_date']} ~ {report['end_date']}", ln=1)
#     pdf.cell(0, 10, f"初始本金: {report['initial_equity']}", ln=1)
#     pdf.cell(0, 10, f"最大回撤率: {report['max_drawdown']:.2%}", ln=1)
#     pdf.cell(0, 10, f"总盈利: {report['total_profit']:.2f}", ln=1)
#     pdf.cell(0, 10, f"利润率: {report['profit_rate']:.2%}", ln=1)
#     pdf.cell(0, 10, f"盈亏比: {report['profit_factor']:.2f}", ln=1)
#     pdf.cell(0, 10, f"交易天数: {report['trade_days']}", ln=1)
#     pdf.cell(0, 10, f"日均交易次数: {report['avg_trades_per_day']:.2f}", ln=1)
#     pdf.cell(0, 10, f"总交易次数: {report['total_trades']}", ln=1)
#     pdf.cell(0, 10, f"盈利次数: {report['winning_trades']}", ln=1)
#     pdf.cell(0, 10, f"盈利占比: {report['win_rate']:.2%}", ln=1)
#     pdf.image(eq_path, x=10, y=pdf.get_y()+5, w=180)
#     pdf.ln(60)
#     pdf.image(dd_path, x=10, y=pdf.get_y()+5, w=180)
#     pdf.output(pdf_path)

# 主流程
if __name__ == "__main__":
    api = TqApi(auth=TqAuth("17817733224", "Tianqin17817733224"))
    klines = api.get_kline_serial("SHFE.cu2201", 60 * 15, data_length=10000)
    df = klines.copy()
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    report = backtest(df)
    eq_path, dd_path = plot_curves(report['equity_curve'], report['drawdown_curve'], "backtest")
    generate_pdf(report, eq_path, dd_path, "backtest_report.pdf")
    api.close()
    print("回测完成，报告已生成：backtest_report.pdf")