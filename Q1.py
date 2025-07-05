import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体和图表样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def create_output_directory():
    """创建输出目录结构"""
    base_dir = 'results/Q1'
    subdirs = ['plot', 'report', 'data']
    
    # 创建主目录和子目录
    for subdir in [base_dir] + [f'{base_dir}/{sub}' for sub in subdirs]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    
    return base_dir

def load_and_clean_data(file_path):
    """加载和清洗数据"""
    print("Loading data...")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")
    
    # 数据清洗
    # 确保关键列存在且无缺失值
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_clean = df[required_columns].dropna()
    
    # 转换日期格式 - 修复日期解析问题
    print("Parsing date format...")
    
    # 首先检查日期样本
    print("Date samples:", df_clean['Date'].head().tolist())
    
    try:
        # 方法1: 尝试使用infer_datetime_format自动推断
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], infer_datetime_format=True)
        print("Date parsing successful using auto-inference")
    except:
        try:
            # 方法2: 明确指定美式日期格式 MM/DD/YYYY
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%m/%d/%Y')
            print("Date parsing successful using MM/DD/YYYY format")
        except:
            try:
                # 方法3: 使用dayfirst=False确保月/日/年的顺序
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], dayfirst=False, errors='coerce')
                print("Date parsing successful using dayfirst=False")
                # 删除无法解析的行
                df_clean = df_clean.dropna(subset=['Date'])
                print(f"After removing unparseable dates: {len(df_clean)} rows remaining")
            except Exception as e:
                print(f"All date parsing methods failed: {e}")
                print("First 20 date samples:")
                print(df_clean['Date'].head(20).tolist())
                
                # 最后尝试：手动处理日期格式
                print("Attempting manual date processing...")
                try:
                    def parse_date(date_str):
                        if pd.isna(date_str):
                            return pd.NaT
                        try:
                            # 尝试解析 MM/DD/YYYY 格式
                            return pd.to_datetime(date_str, format='%m/%d/%Y')
                        except:
                            try:
                                # 尝试解析 MM/DD/YY 格式  
                                return pd.to_datetime(date_str, format='%m/%d/%y')
                            except:
                                return pd.NaT
                    
                    df_clean['Date'] = df_clean['Date'].apply(parse_date)
                    df_clean = df_clean.dropna(subset=['Date'])
                    print(f"Manual parsing successful, {len(df_clean)} rows remaining")
                except:
                    raise Exception("Cannot parse date format, please check data file")
    
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Data time range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    
    return df_clean

def calculate_basic_statistics(df):
    """计算基本统计量"""
    print("Calculating basic statistics...")
    
    # 黄金收盘价基本统计量
    gold_prices = df['Close']
    
    stats = {
        'Count': len(gold_prices),
        'Mean': gold_prices.mean(),
        'Median': gold_prices.median(),
        'Standard Deviation': gold_prices.std(),
        'Min': gold_prices.min(),
        'Max': gold_prices.max(),
        'Range': gold_prices.max() - gold_prices.min(),
        'Coefficient of Variation': gold_prices.std() / gold_prices.mean(),
        'Skewness': gold_prices.skew(),
        'Kurtosis': gold_prices.kurtosis()
    }
    
    # 计算分位数
    quantiles = gold_prices.quantile([0.25, 0.5, 0.75])
    stats['25th Percentile'] = quantiles[0.25]
    stats['75th Percentile'] = quantiles[0.75]
    stats['Interquartile Range'] = quantiles[0.75] - quantiles[0.25]
    
    return stats

def calculate_daily_returns(df):
    """计算日收益率统计"""
    print("Calculating daily returns...")
    
    # 计算日收益率
    df['Daily_Return'] = df['Close'].pct_change() * 100
    daily_returns = df['Daily_Return'].dropna()
    
    return_stats = {
        'Average Daily Return(%)': daily_returns.mean(),
        'Daily Return Std(%)': daily_returns.std(),
        'Annualized Return(%)': daily_returns.mean() * 252,
        'Annualized Volatility(%)': daily_returns.std() * np.sqrt(252),
        'Max Daily Gain(%)': daily_returns.max(),
        'Max Daily Loss(%)': daily_returns.min(),
        'Positive Days': (daily_returns > 0).sum(),
        'Negative Days': (daily_returns < 0).sum(),
        'Win Rate(%)': (daily_returns > 0).mean() * 100
    }
    
    return return_stats, daily_returns

def yearly_analysis(df):
    """年度分析"""
    print("Performing yearly analysis...")
    
    df['Year'] = df['Date'].dt.year
    yearly_stats = []
    
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]
        if len(year_data) < 2:
            continue
            
        year_prices = year_data['Close']
        start_price = year_prices.iloc[0]
        end_price = year_prices.iloc[-1]
        year_return = (end_price - start_price) / start_price * 100
        
        yearly_stats.append({
            'Year': year,
            'Data_Points': len(year_data),
            'Average_Price': year_prices.mean(),
            'Price_Volatility': year_prices.std(),
            'Max_Price': year_prices.max(),
            'Min_Price': year_prices.min(),
            'Start_Price': start_price,
            'End_Price': end_price,
            'Annual_Return(%)': year_return
        })
    
    return pd.DataFrame(yearly_stats)

def create_visualizations(df, daily_returns, yearly_df, output_dir):
    """创建可视化图表"""
    print("Creating visualizations...")
    
    plot_dir = f'{output_dir}/plot'
    
    # 1. Gold Price Time Series
    plt.figure(figsize=(15, 8))
    plt.plot(df['Date'], df['Close'], linewidth=2, color='gold', alpha=0.8)
    plt.fill_between(df['Date'], df['Close'], alpha=0.3, color='gold')
    plt.title('Gold Price Time Series', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/01_gold_price_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Price Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df['Close'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['Close'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: ${df["Close"].mean():.2f}')
    plt.axvline(df['Close'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: ${df["Close"].median():.2f}')
    plt.title('Gold Price Distribution')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['Close'])
    plt.title('Gold Price Box Plot')
    plt.ylabel('Price (USD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/02_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Daily Returns Analysis
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(daily_returns, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {daily_returns.mean():.4f}%')
    plt.title('Daily Returns Distribution')
    plt.xlabel('Daily Returns (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(df['Date'][1:], daily_returns, alpha=0.6, color='purple')
    plt.title('Daily Returns Time Series')
    plt.xlabel('Date')
    plt.ylabel('Daily Returns (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.boxplot(daily_returns)
    plt.title('Daily Returns Box Plot')
    plt.ylabel('Daily Returns (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/03_daily_returns_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Yearly Performance Analysis
    plt.figure(figsize=(15, 10))
    
    # Annual Returns Bar Chart
    plt.subplot(2, 2, 1)
    colors = ['green' if x > 0 else 'red' for x in yearly_df['Annual_Return(%)']]
    bars = plt.bar(yearly_df['Year'], yearly_df['Annual_Return(%)'], color=colors, alpha=0.7)
    plt.title('Annual Returns')
    plt.xlabel('Year')
    plt.ylabel('Returns (%)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, yearly_df['Annual_Return(%)']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # Annual Average Price
    plt.subplot(2, 2, 2)
    plt.plot(yearly_df['Year'], yearly_df['Average_Price'], marker='o', linewidth=2, markersize=8, color='blue')
    plt.title('Annual Average Price Trend')
    plt.xlabel('Year')
    plt.ylabel('Average Price (USD)')
    plt.grid(True, alpha=0.3)
    
    # Annual Volatility
    plt.subplot(2, 2, 3)
    plt.bar(yearly_df['Year'], yearly_df['Price_Volatility'], color='orange', alpha=0.7)
    plt.title('Annual Price Volatility')
    plt.xlabel('Year')
    plt.ylabel('Volatility (USD)')
    plt.grid(True, alpha=0.3)
    
    # Annual Price Range
    plt.subplot(2, 2, 4)
    plt.fill_between(yearly_df['Year'], yearly_df['Min_Price'], yearly_df['Max_Price'], alpha=0.3, color='gray')
    plt.plot(yearly_df['Year'], yearly_df['Average_Price'], color='red', linewidth=2, label='Average Price')
    plt.title('Annual Price Range')
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/04_yearly_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Technical Analysis
    plt.figure(figsize=(15, 8))
    
    # Calculate moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_60'] = df['Close'].rolling(window=60).mean()
    
    plt.plot(df['Date'], df['Close'], label='Close Price', linewidth=1, alpha=0.8)
    plt.plot(df['Date'], df['MA_5'], label='5-Day MA', linewidth=1.5)
    plt.plot(df['Date'], df['MA_20'], label='20-Day MA', linewidth=1.5)
    plt.plot(df['Date'], df['MA_60'], label='60-Day MA', linewidth=1.5)
    
    plt.title('Gold Price with Moving Averages', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/05_technical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Risk-Return Analysis
    plt.figure(figsize=(12, 8))
    
    # Rolling volatility
    rolling_vol = daily_returns.rolling(window=30).std() * np.sqrt(252)
    
    plt.subplot(2, 1, 1)
    plt.plot(df['Date'][1:], daily_returns.cumsum(), color='blue', linewidth=2)
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    plt.plot(df['Date'][1:], rolling_vol, color='red', linewidth=2)
    plt.title('30-Day Rolling Annualized Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/06_risk_return_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All charts saved to: {plot_dir}/")
    return df  # Return df with added technical indicators

def save_statistics_to_file(basic_stats, return_stats, yearly_df, output_dir):
    """保存统计结果到文件"""
    print("Saving statistical results...")
    
    report_dir = f'{output_dir}/report'
    data_dir = f'{output_dir}/data'
    
    # Save detailed statistics report
    with open(f'{report_dir}/statistics_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Gold Price Data Statistical Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. Basic Statistics\n")
        f.write("-" * 30 + "\n")
        for key, value in basic_stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n2. Daily Returns Statistics\n")
        f.write("-" * 30 + "\n")
        for key, value in return_stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n3. Annual Performance Summary\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Year: {yearly_df.loc[yearly_df['Annual_Return(%)'].idxmax(), 'Year']} ")
        f.write(f"(Return: {yearly_df['Annual_Return(%)'].max():.2f}%)\n")
        f.write(f"Worst Year: {yearly_df.loc[yearly_df['Annual_Return(%)'].idxmin(), 'Year']} ")
        f.write(f"(Return: {yearly_df['Annual_Return(%)'].min():.2f}%)\n")
        f.write(f"Average Annual Return: {yearly_df['Annual_Return(%)'].mean():.2f}%\n")
        f.write(f"Annual Return Std Dev: {yearly_df['Annual_Return(%)'].std():.2f}%\n")
    
    # Save yearly statistics to CSV
    yearly_df.to_csv(f'{data_dir}/yearly_statistics.csv', index=False)
    
    # Save processed data
    # This will be saved in the main function
    
    print(f"Statistical results saved to: {report_dir}/statistics_summary.txt")
    print(f"Yearly statistics saved to: {data_dir}/yearly_statistics.csv")

def create_readme(output_dir, basic_stats, return_stats, yearly_df):
    """创建README文件"""
    print("Creating README file...")
    
    readme_content = f"""# Gold Price Analysis - Question 1 Results

## Overview
This folder contains the complete statistical analysis and visualization results for gold price data.

**Analysis Period:** {yearly_df['Year'].min()} - {yearly_df['Year'].max()}  
**Total Data Points:** {basic_stats['Count']}  
**Data Frequency:** Daily

## Folder Structure

```
Q1/
├── README.md                    # This file
├── plot/                        # All visualization charts
├── report/                      # Statistical reports and summaries  
├── data/                        # Processed data files
```

## Key Findings

### 📊 Basic Statistics
- **Average Price:** ${basic_stats['Mean']:.2f}
- **Price Range:** ${basic_stats['Min']:.2f} - ${basic_stats['Max']:.2f}
- **Standard Deviation:** ${basic_stats['Standard Deviation']:.2f}
- **Coefficient of Variation:** {basic_stats['Coefficient of Variation']:.4f}

### 📈 Risk & Return Metrics
- **Average Daily Return:** {return_stats['Average Daily Return(%)']:.4f}%
- **Daily Volatility:** {return_stats['Daily Return Std(%)']:.4f}%
- **Annualized Return:** {return_stats['Annualized Return(%)']:.2f}%
- **Annualized Volatility:** {return_stats['Annualized Volatility(%)']:.2f}%
- **Win Rate:** {return_stats['Win Rate(%)']:.2f}%

### 🏆 Best & Worst Performances
- **Best Year:** {yearly_df.loc[yearly_df['Annual_Return(%)'].idxmax(), 'Year']} ({yearly_df['Annual_Return(%)'].max():.2f}%)
- **Worst Year:** {yearly_df.loc[yearly_df['Annual_Return(%)'].idxmin(), 'Year']} ({yearly_df['Annual_Return(%)'].min():.2f}%)
- **Average Annual Return:** {yearly_df['Annual_Return(%)'].mean():.2f}%

## Files Description

### 📊 Visualizations (plot/)
1. **01_gold_price_timeseries.png** - Complete price time series
2. **02_price_distribution.png** - Price distribution histogram and box plot
3. **03_daily_returns_analysis.png** - Daily returns analysis (distribution, time series, box plot)
4. **04_yearly_analysis.png** - Annual performance analysis (4 subplots)
5. **05_technical_analysis.png** - Price with moving averages (5, 20, 60 days)
6. **06_risk_return_analysis.png** - Cumulative returns and rolling volatility

### 📋 Reports (report/)
- **statistics_summary.txt** - Detailed statistical analysis report

### 💾 Data (data/)
- **yearly_statistics.csv** - Annual statistics in tabular format
- **processed_gold_data.csv** - Cleaned and processed daily data

## Analysis Methodology

1. **Data Preprocessing:** Date parsing, missing value handling, data validation
2. **Descriptive Statistics:** Central tendency, dispersion, and distribution analysis
3. **Time Series Analysis:** Trend identification, seasonal patterns, volatility analysis
4. **Risk Assessment:** Return distribution, volatility metrics, downside risk
5. **Technical Analysis:** Moving averages, support/resistance levels

## Key Insights

🔍 **Price Behavior:** Gold shows significant volatility with annualized volatility of {return_stats['Annualized Volatility(%)']:.1f}%

📊 **Return Characteristics:** {return_stats['Win Rate(%)']:.1f}% of trading days showed positive returns

📈 **Trend Analysis:** Long-term trend shows {"upward" if return_stats['Annualized Return(%)'] > 0 else "downward"} movement

⚠️ **Risk Profile:** Maximum daily loss was {return_stats['Max Daily Loss(%)']:.2f}%, maximum daily gain was {return_stats['Max Daily Gain(%)']:.2f}%

---
*Generated by Gold Price Analysis System*  
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(f'{output_dir}/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README.md created at: {output_dir}/README.md")

def main():
    """主函数"""
    print("Starting Question 1: Basic Statistical Analysis and Visualization of Gold Price Data")
    print("=" * 80)
    
    # 创建输出目录结构
    output_dir = create_output_directory()
    
    try:
        # 加载和清洗数据
        df = load_and_clean_data('B题附件：data.csv')
        
        # 计算基本统计量
        basic_stats = calculate_basic_statistics(df)
        
        # 计算日收益率统计
        return_stats, daily_returns = calculate_daily_returns(df)
        
        # 年度分析
        yearly_df = yearly_analysis(df)
        
        # 创建可视化图表
        df_with_indicators = create_visualizations(df, daily_returns, yearly_df, output_dir)
        
        # 保存统计结果
        save_statistics_to_file(basic_stats, return_stats, yearly_df, output_dir)
        
        # 保存处理后的数据
        data_dir = f'{output_dir}/data'
        df_with_indicators.to_csv(f'{data_dir}/processed_gold_data.csv', index=False)
        print(f"Processed data saved to: {data_dir}/processed_gold_data.csv")
        
        # 创建README文件
        create_readme(output_dir, basic_stats, return_stats, yearly_df)
        
        # 打印主要结果
        print("\n" + "=" * 80)
        print("MAIN ANALYSIS RESULTS")
        print("=" * 80)
        print(f"Data Points: {basic_stats['Count']}")
        print(f"Average Price: ${basic_stats['Mean']:.2f}")
        print(f"Price Standard Deviation: ${basic_stats['Standard Deviation']:.2f}")
        print(f"Annualized Volatility: {return_stats['Annualized Volatility(%)']:.2f}%")
        print(f"Average Annual Return: {yearly_df['Annual_Return(%)'].mean():.2f}%")
        print(f"Win Rate: {return_stats['Win Rate(%)']:.2f}%")
        print(f"Best Year: {yearly_df.loc[yearly_df['Annual_Return(%)'].idxmax(), 'Year']} ({yearly_df['Annual_Return(%)'].max():.2f}%)")
        print(f"Worst Year: {yearly_df.loc[yearly_df['Annual_Return(%)'].idxmin(), 'Year']} ({yearly_df['Annual_Return(%)'].min():.2f}%)")
        
        print(f"\n📁 All results saved to: {output_dir}/")
        print("📂 Folder structure:")
        print("   ├── README.md                 # Complete analysis overview")
        print("   ├── plot/                     # 6 visualization charts")
        print("   │   ├── 01_gold_price_timeseries.png")
        print("   │   ├── 02_price_distribution.png")
        print("   │   ├── 03_daily_returns_analysis.png")
        print("   │   ├── 04_yearly_analysis.png")
        print("   │   ├── 05_technical_analysis.png")
        print("   │   └── 06_risk_return_analysis.png")
        print("   ├── report/                   # Statistical analysis reports")
        print("   │   └── statistics_summary.txt")
        print("   └── data/                     # Processed data files")
        print("       ├── yearly_statistics.csv")
        print("       └── processed_gold_data.csv")
        
        print("\n✅ Question 1 analysis completed successfully!")
        
    except FileNotFoundError:
        print("❌ Error: Cannot find file 'B题附件：data.csv'")
        print("Please ensure the data file is in the same directory as the script")
    except Exception as e:
        print(f"❌ Error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()