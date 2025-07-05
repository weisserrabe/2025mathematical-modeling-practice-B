import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
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
    base_dir = 'results/Q2'
    subdirs = ['plot', 'report', 'data']
    
    # 创建主目录和子目录
    for subdir in [base_dir] + [f'{base_dir}/{sub}' for sub in subdirs]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    
    return base_dir

def load_data():
    """加载数据"""
    print("Loading data for correlation analysis...")
    
    # 直接加载原始数据以获取所有市场因素
    try:
        df = pd.read_csv('B题附件：data.csv')
        print("Loaded original data file")
        
        # 处理日期格式
        print("Processing date format...")
        try:
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
            print("Date parsing successful using auto-inference")
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
                print("Date parsing successful using MM/DD/YYYY format")
            except:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')
                df = df.dropna(subset=['Date'])
                print("Date parsing successful using dayfirst=False")
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 显示可用的列
        print("Available columns:", df.columns.tolist()[:15], "..." if len(df.columns) > 15 else "")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def check_available_columns(df):
    """检查数据中可用的列"""
    print("\n" + "="*50)
    print("AVAILABLE DATA COLUMNS")
    print("="*50)
    
    # 按类别分组显示列名
    all_columns = df.columns.tolist()
    
    print(f"Total columns: {len(all_columns)}")
    print("\nAll available columns:")
    for i, col in enumerate(all_columns):
        print(f"{i+1:2d}. {col}")
    
    # 尝试识别可能的市场因素列
    potential_factors = []
    factor_keywords = ['SP', 'DJ', 'EU', 'OF', 'OS', 'USD', 'PLT', 'PLD', 'USB', 'GDX', 'USO']
    
    for col in all_columns:
        for keyword in factor_keywords:
            if keyword.lower() in col.lower():
                potential_factors.append(col)
                break
    
    if potential_factors:
        print(f"\nPotential market factor columns ({len(potential_factors)}):")
        for i, col in enumerate(potential_factors):
            print(f"{i+1:2d}. {col}")
    
    return potential_factors

def define_factors():
    """定义影响因素"""
    factors = {
        'SP500_Index': 'SP_close',           # S&P 500 Index
        'Dow_Jones_Index': 'DJ_close',       # Dow Jones Index  
        'EUR_USD_Rate': 'EU_Price',          # EUR/USD Exchange Rate
        'Oil_Price': 'OF_Price',             # Oil Futures Price
        'Silver_Price': 'OS_Price',          # Silver Price
        'USD_Index': 'USDI_Price',           # US Dollar Index
        'Platinum_Price': 'PLT_Price',       # Platinum Price
        'Palladium_Price': 'PLD_Price',      # Palladium Price
        'US_Bonds': 'USB_Price',             # US Treasury Bonds
        'Gold_Miners_ETF': 'GDX_Close',      # Gold Miners ETF (GDX)
        'Oil_ETF': 'USO_Close'               # Oil ETF (USO)
    }
    
    return factors
    """定义影响因素"""
    factors = {
        'SP500_Index': 'SP_close',           # S&P 500 Index
        'Dow_Jones_Index': 'DJ_close',       # Dow Jones Index  
        'EUR_USD_Rate': 'EU_Price',          # EUR/USD Exchange Rate
        'Oil_Price': 'OF_Price',             # Oil Futures Price
        'Silver_Price': 'OS_Price',          # Silver Price
        'USD_Index': 'USDI_Price',           # US Dollar Index
        'Platinum_Price': 'PLT_Price',       # Platinum Price
        'Palladium_Price': 'PLD_Price',      # Palladium Price
        'US_Bonds': 'USB_Price',             # US Treasury Bonds
        'Gold_Miners_ETF': 'GDX_Close',      # Gold Miners ETF (GDX)
        'Oil_ETF': 'USO_Close'               # Oil ETF (USO)
    }
    
    return factors

def calculate_correlations(df, factors):
    """计算相关性"""
    print("Calculating correlations...")
    
    # 确保所有需要的列都存在
    available_factors = {}
    for name, column in factors.items():
        if column in df.columns:
            available_factors[name] = column
        else:
            print(f"Warning: {column} not found in data")
    
    if len(available_factors) == 0:
        raise ValueError("No valid factors found in data! Please check column names.")
    
    print(f"Found {len(available_factors)} valid factors: {list(available_factors.keys())}")
    
    # 筛选有效数据
    factor_columns = ['Close'] + list(available_factors.values())
    correlation_data = df[factor_columns].dropna()
    
    print(f"Valid data points for correlation analysis: {len(correlation_data)}")
    
    # 计算相关系数矩阵
    correlation_matrix = correlation_data.corr()
    
    # 提取与黄金价格的相关性
    gold_correlations = correlation_matrix['Close'].drop('Close')
    
    # 创建详细的相关性结果
    correlation_results = []
    for factor_name, column in available_factors.items():
        if column in gold_correlations.index:
            corr_value = gold_correlations[column]
            
            # 计算统计显著性
            factor_data = correlation_data[['Close', column]].dropna()
            if len(factor_data) > 2:  # 需要至少3个数据点
                corr_coef, p_value = stats.pearsonr(factor_data['Close'], factor_data[column])
            else:
                corr_coef, p_value = corr_value, 1.0
            
            # 相关性强度分类
            abs_corr = abs(corr_value)
            if abs_corr >= 0.7:
                strength = 'Very Strong'
            elif abs_corr >= 0.5:
                strength = 'Strong'
            elif abs_corr >= 0.3:
                strength = 'Moderate'
            elif abs_corr >= 0.1:
                strength = 'Weak'
            else:
                strength = 'Very Weak'
            
            direction = 'Positive' if corr_value > 0 else 'Negative'
            significance = 'Significant' if p_value < 0.05 else 'Not Significant'
            
            correlation_results.append({
                'Factor': factor_name,
                'Column': column,
                'Correlation': corr_value,
                'Abs_Correlation': abs_corr,
                'P_Value': p_value,
                'Direction': direction,
                'Strength': strength,
                'Significance': significance
            })
    
    # 转换为DataFrame并按相关性强度排序
    if len(correlation_results) == 0:
        # 如果没有任何结果，创建一个空的DataFrame
        correlation_df = pd.DataFrame(columns=['Factor', 'Column', 'Correlation', 'Abs_Correlation', 
                                             'P_Value', 'Direction', 'Strength', 'Significance'])
        print("Warning: No correlation results generated!")
    else:
        correlation_df = pd.DataFrame(correlation_results)
        correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)
    
    return correlation_matrix, correlation_df, correlation_data, available_factors

def create_visualizations(correlation_matrix, correlation_df, correlation_data, available_factors, output_dir, original_df):
    """创建可视化图表"""
    print("Creating correlation visualizations...")
    
    plot_dir = f'{output_dir}/plot'
    
    # 1. 相关性热力图
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix: Gold Price vs Market Factors', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 黄金价格相关性条形图
    plt.figure(figsize=(14, 8))
    
    # 按相关性排序
    sorted_factors = correlation_df.sort_values('Correlation', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in sorted_factors['Correlation']]
    
    bars = plt.barh(range(len(sorted_factors)), sorted_factors['Correlation'], color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_factors)), sorted_factors['Factor'])
    plt.xlabel('Correlation Coefficient')
    plt.title('Gold Price Correlation with Market Factors', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, sorted_factors['Correlation'])):
        plt.text(value + (0.02 if value >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/02_correlation_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 散点图矩阵（top 6 相关性）
    top_factors = correlation_df.head(6)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (_, factor_info) in enumerate(top_factors.iterrows()):
        if i >= 6:  # 安全检查
            break
            
        factor_name = factor_info['Factor']
        column = factor_info['Column']
        corr_value = factor_info['Correlation']
        
        x_data = correlation_data[column]
        y_data = correlation_data['Close']
        
        axes[i].scatter(x_data, y_data, alpha=0.6, s=20)
        
        # 添加趋势线
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        axes[i].plot(x_data, p(x_data), "r--", alpha=0.8)
        
        axes[i].set_xlabel(factor_name)
        axes[i].set_ylabel('Gold Price (USD)')
        axes[i].set_title(f'{factor_name}\nCorr: {corr_value:.3f}')
        axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(top_factors), 6):
        axes[i].set_visible(False)
    
    plt.suptitle('Gold Price vs Top 6 Correlated Factors', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/03_scatter_plots_top6.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 时间序列对比（标准化）
    plt.figure(figsize=(16, 10))
    
    # 选择top 4因素进行时间序列对比
    top_4_factors = correlation_df.head(4)
    
    if len(top_4_factors) > 0:
        # 创建包含日期的数据集
        # 获取与correlation_data相同索引的日期
        dates = original_df.loc[correlation_data.index, 'Date'].reset_index(drop=True)
        
        # 标准化数据
        scaler = StandardScaler()
        numeric_columns = ['Close'] + [f['Column'] for _, f in top_4_factors.iterrows()]
        plot_data = correlation_data[numeric_columns].copy()
        plot_data_scaled = pd.DataFrame(
            scaler.fit_transform(plot_data),
            columns=numeric_columns,
            index=plot_data.index
        )
        
        # 绘制标准化时间序列
        plt.plot(dates, plot_data_scaled['Close'], label='Gold Price', linewidth=2, color='gold')
        
        colors = ['red', 'blue', 'green', 'purple']
        for i, (_, factor_info) in enumerate(top_4_factors.iterrows()):
            if i >= 4:  # 安全检查
                break
                
            factor_name = factor_info['Factor']
            column = factor_info['Column']
            corr_value = factor_info['Correlation']
            
            plt.plot(dates, plot_data_scaled[column], 
                    label=f'{factor_name} (r={corr_value:.3f})', 
                    linewidth=1.5, 
                    color=colors[i],
                    alpha=0.8)
        
        plt.title('Normalized Time Series: Gold Price vs Top Correlated Factors', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Standardized Values')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'No factors available for time series analysis', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title('Time Series Analysis - No Data Available')
    
    plt.savefig(f'{plot_dir}/04_normalized_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 相关性强度分布
    plt.figure(figsize=(12, 8))
    
    # 相关性强度分布饼图
    plt.subplot(2, 2, 1)
    strength_counts = correlation_df['Strength'].value_counts()
    plt.pie(strength_counts.values, labels=strength_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Correlation Strength')
    
    # 相关性方向分布
    plt.subplot(2, 2, 2)
    direction_counts = correlation_df['Direction'].value_counts()
    colors = ['green', 'red']
    plt.pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Distribution of Correlation Direction')
    
    # 相关性系数分布直方图
    plt.subplot(2, 2, 3)
    plt.hist(correlation_df['Correlation'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(correlation_df['Correlation'].mean(), color='red', linestyle='--', 
                label=f'Mean: {correlation_df["Correlation"].mean():.3f}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Distribution of Correlation Coefficients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 统计显著性
    plt.subplot(2, 2, 4)
    significance_counts = correlation_df['Significance'].value_counts()
    plt.pie(significance_counts.values, labels=significance_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Statistical Significance Distribution')
    
    plt.suptitle('Correlation Analysis Summary Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/05_correlation_summary_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 滚动相关性分析（对于top 3因素）
    plt.figure(figsize=(16, 12))
    
    top_3_factors = correlation_df.head(3)
    window = 252  # 一年的交易日
    
    if len(top_3_factors) > 0:
        # 获取与correlation_data相同索引的日期
        dates = original_df.loc[correlation_data.index, 'Date'].reset_index(drop=True)
        
        for i, (_, factor_info) in enumerate(top_3_factors.iterrows()):
            factor_name = factor_info['Factor']
            column = factor_info['Column']
            
            plt.subplot(3, 1, i+1)
            
            # 计算滚动相关性
            rolling_corr = correlation_data['Close'].rolling(window=window).corr(correlation_data[column])
            
            plt.plot(dates, rolling_corr, linewidth=2, label=f'Rolling Correlation')
            plt.axhline(y=factor_info['Correlation'], color='red', linestyle='--', 
                       label=f'Overall Correlation: {factor_info["Correlation"]:.3f}')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.title(f'Rolling Correlation: Gold Price vs {factor_name} ({window}-day window)')
            plt.xlabel('Date')
            plt.ylabel('Correlation Coefficient')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No factors available for rolling correlation analysis', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title('Rolling Correlation Analysis - No Data Available')
    
    plt.suptitle('Rolling Correlation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/06_rolling_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All correlation visualizations saved to: {plot_dir}/")

def save_analysis_results(correlation_matrix, correlation_df, output_dir):
    """保存分析结果"""
    print("Saving correlation analysis results...")
    
    report_dir = f'{output_dir}/report'
    data_dir = f'{output_dir}/data'
    
    # 保存详细分析报告
    with open(f'{report_dir}/correlation_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GOLD PRICE CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        if len(correlation_df) == 0:
            f.write("ERROR: No valid market factors found for correlation analysis!\n")
            f.write("Please check if the data file contains the required columns.\n\n")
            f.write("Expected columns:\n")
            expected_factors = define_factors()
            for name, column in expected_factors.items():
                f.write(f"- {name}: {column}\n")
            return
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total factors analyzed: {len(correlation_df)}\n")
        f.write(f"Strong correlations (|r| > 0.5): {len(correlation_df[correlation_df['Abs_Correlation'] > 0.5])}\n")
        f.write(f"Positive correlations: {len(correlation_df[correlation_df['Correlation'] > 0])}\n")
        f.write(f"Negative correlations: {len(correlation_df[correlation_df['Correlation'] < 0])}\n")
        f.write(f"Statistically significant (p < 0.05): {len(correlation_df[correlation_df['Significance'] == 'Significant'])}\n\n")
        
        f.write("TOP CORRELATIONS\n")
        f.write("-" * 40 + "\n")
        for i, (_, row) in enumerate(correlation_df.head(10).iterrows(), 1):
            f.write(f"{i:2d}. {row['Factor']:<20} | r = {row['Correlation']:>7.4f} | {row['Strength']:<12} | {row['Significance']}\n")
        
        f.write("\n\nDETAILED CORRELATION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Factor':<25} {'Correlation':<12} {'P-Value':<10} {'Strength':<15} {'Direction':<10} {'Significance':<15}\n")
        f.write("-" * 100 + "\n")
        
        for _, row in correlation_df.iterrows():
            f.write(f"{row['Factor']:<25} {row['Correlation']:<12.4f} {row['P_Value']:<10.4f} "
                   f"{row['Strength']:<15} {row['Direction']:<10} {row['Significance']:<15}\n")
        
        f.write("\n\nKEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # 最强正相关
        strongest_pos = correlation_df[correlation_df['Correlation'] > 0].iloc[0] if len(correlation_df[correlation_df['Correlation'] > 0]) > 0 else None
        if strongest_pos is not None:
            f.write(f"Strongest positive correlation: {strongest_pos['Factor']} (r = {strongest_pos['Correlation']:.4f})\n")
        
        # 最强负相关
        strongest_neg = correlation_df[correlation_df['Correlation'] < 0].iloc[0] if len(correlation_df[correlation_df['Correlation'] < 0]) > 0 else None
        if strongest_neg is not None:
            f.write(f"Strongest negative correlation: {strongest_neg['Factor']} (r = {strongest_neg['Correlation']:.4f})\n")
        
        f.write(f"\nAverage correlation magnitude: {correlation_df['Abs_Correlation'].mean():.4f}\n")
        f.write(f"Standard deviation of correlations: {correlation_df['Correlation'].std():.4f}\n")
        
        # 相关性解释
        f.write("\n\nCORRELATION INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write("Very Strong (|r| ≥ 0.7): Variables move almost in perfect sync\n")
        f.write("Strong (0.5 ≤ |r| < 0.7): Strong linear relationship\n")
        f.write("Moderate (0.3 ≤ |r| < 0.5): Moderate linear relationship\n")
        f.write("Weak (0.1 ≤ |r| < 0.3): Weak linear relationship\n")
        f.write("Very Weak (|r| < 0.1): Little to no linear relationship\n")
    
    # 保存相关性矩阵
    if len(correlation_matrix) > 0:
        correlation_matrix.to_csv(f'{data_dir}/correlation_matrix.csv')
    
    # 保存详细相关性结果
    correlation_df.to_csv(f'{data_dir}/correlation_results.csv', index=False)
    
    print(f"Correlation analysis report saved to: {report_dir}/correlation_analysis_report.txt")
    if len(correlation_matrix) > 0:
        print(f"Correlation matrix saved to: {data_dir}/correlation_matrix.csv")
    print(f"Detailed results saved to: {data_dir}/correlation_results.csv")

def create_readme(correlation_df, output_dir):
    """创建README文件"""
    print("Creating README file...")
    
    # 计算统计信息
    strong_correlations = correlation_df[correlation_df['Abs_Correlation'] > 0.5]
    significant_correlations = correlation_df[correlation_df['Significance'] == 'Significant']
    
    readme_content = f"""# Gold Price Correlation Analysis - Question 2 Results

## Overview
This folder contains comprehensive correlation analysis between gold prices and major market factors.

**Analysis Method:** Pearson Correlation Coefficient  
**Factors Analyzed:** {len(correlation_df)}  
**Data Points:** Based on Q1 processed data  
**Significance Level:** α = 0.05

## Folder Structure

```
Q2/
├── README.md                           # This file
├── plot/                              # Correlation visualizations
├── report/                            # Analysis reports
└── data/                              # Correlation data and matrices
```

## Key Findings

### 🏆 Top 5 Correlations
"""
    
    for i, (_, row) in enumerate(correlation_df.head(5).iterrows(), 1):
        direction_emoji = "📈" if row['Correlation'] > 0 else "📉"
        readme_content += f"{i}. **{row['Factor']}** {direction_emoji} r = {row['Correlation']:.4f} ({row['Strength']})\n"
    
    readme_content += f"""

### 📊 Correlation Statistics
- **Strong Correlations (|r| > 0.5):** {len(strong_correlations)} factors
- **Positive Correlations:** {len(correlation_df[correlation_df['Correlation'] > 0])} factors  
- **Negative Correlations:** {len(correlation_df[correlation_df['Correlation'] < 0])} factors
- **Statistically Significant:** {len(significant_correlations)} factors
- **Average |Correlation|:** {correlation_df['Abs_Correlation'].mean():.4f}

### 🎯 Market Insights

#### Strongest Positive Drivers"""
    
    positive_corrs = correlation_df[correlation_df['Correlation'] > 0].head(3)
    for _, row in positive_corrs.iterrows():
        readme_content += f"\n- **{row['Factor']}** (r = {row['Correlation']:.3f}): Strong positive relationship suggests these move together with gold"
    
    readme_content += "\n\n#### Strongest Negative Drivers"
    
    negative_corrs = correlation_df[correlation_df['Correlation'] < 0].head(3)
    for _, row in negative_corrs.iterrows():
        readme_content += f"\n- **{row['Factor']}** (r = {row['Correlation']:.3f}): Strong negative relationship suggests inverse movement"
    
    readme_content += f"""

## Files Description

### 📊 Visualizations (plot/)
1. **01_correlation_heatmap.png** - Full correlation matrix heatmap
2. **02_correlation_barplot.png** - Gold price correlations ranked
3. **03_scatter_plots_top6.png** - Scatter plots for top 6 factors
4. **04_normalized_timeseries.png** - Time series comparison (standardized)
5. **05_correlation_summary_stats.png** - Distribution analysis (4 subplots)
6. **06_rolling_correlations.png** - Rolling correlation analysis

### 📋 Reports (report/)
- **correlation_analysis_report.txt** - Comprehensive analysis report

### 💾 Data (data/)
- **correlation_matrix.csv** - Full correlation matrix
- **correlation_results.csv** - Detailed correlation statistics

## Methodology

### Data Processing
1. **Data Source:** Q1 processed gold price data + market factors
2. **Missing Values:** Listwise deletion for correlation pairs
3. **Timeframe:** Complete overlap period of all factors

### Statistical Analysis
1. **Pearson Correlation:** Linear relationship measurement
2. **Significance Testing:** p-value < 0.05 threshold
3. **Rolling Analysis:** 252-day rolling correlations
4. **Standardization:** Z-score normalization for time series comparison

### Correlation Strength Classification
- **Very Strong:** |r| ≥ 0.7
- **Strong:** 0.5 ≤ |r| < 0.7  
- **Moderate:** 0.3 ≤ |r| < 0.5
- **Weak:** 0.1 ≤ |r| < 0.3
- **Very Weak:** |r| < 0.1

## Economic Interpretation

### Risk-On vs Risk-Off Dynamics
The correlation patterns reveal gold's role in different market conditions:

**Risk-Off Assets (Positive Correlation):**
- Precious metals (Silver, Platinum) show strong positive correlations
- These assets tend to rise together during uncertainty

**Risk-On Assets (Negative Correlation):**  
- Stock indices (S&P 500, Dow Jones) show negative correlations
- When markets are optimistic, gold demand typically decreases

**Currency Effects:**
- USD Index typically shows strong negative correlation
- EUR/USD movements affect gold's attractiveness to different currencies

### Investment Implications

🔍 **Portfolio Diversification:** Factors with low/negative correlations provide diversification benefits

📈 **Trend Following:** High positive correlations indicate potential momentum strategies

⚖️ **Hedging Strategies:** Strong negative correlations suggest effective hedging pairs

---
*Generated by Gold Price Correlation Analysis System*  
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(f'{output_dir}/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README.md created at: {output_dir}/README.md")

def main():
    """主函数"""
    print("Starting Question 2: Market Factors Correlation Analysis")
    print("=" * 80)
    
    # 创建输出目录结构
    output_dir = create_output_directory()
    
    try:
        # 加载数据
        df = load_data()
        
        # 检查可用列名
        potential_factors = check_available_columns(df)
        
        # 定义影响因素
        factors = define_factors()
        
        # 计算相关性
        correlation_matrix, correlation_df, correlation_data, available_factors = calculate_correlations(df, factors)
        
        # 检查是否有有效的相关性结果
        if len(correlation_df) == 0:
            print("❌ No valid correlations found!")
            print("Please check if the data file contains the required market factor columns.")
            print("Expected columns:", list(factors.values()))
            return
        
        print(f"✅ Successfully calculated correlations for {len(correlation_df)} factors")
        
        # 创建可视化
        if len(correlation_df) > 0:
            create_visualizations(correlation_matrix, correlation_df, correlation_data, available_factors, output_dir, df)
        else:
            print("⚠️  Skipping visualizations due to insufficient data")
        
        # 保存分析结果
        save_analysis_results(correlation_matrix, correlation_df, output_dir)
        
        # 创建README
        if len(correlation_df) > 0:
            create_readme(correlation_df, output_dir)
        else:
            print("⚠️  Skipping README creation due to insufficient data")
        
        # 打印主要结果
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS RESULTS")
        print("=" * 80)
        
        if len(correlation_df) > 0:
            print(f"Factors analyzed: {len(correlation_df)}")
            print(f"Strong correlations (|r| > 0.5): {len(correlation_df[correlation_df['Abs_Correlation'] > 0.5])}")
            print(f"Significant correlations: {len(correlation_df[correlation_df['Significance'] == 'Significant'])}")
            
            print(f"\n📊 TOP 5 CORRELATIONS:")
            for i, (_, row) in enumerate(correlation_df.head(5).iterrows(), 1):
                print(f"{i}. {row['Factor']:<25} r = {row['Correlation']:>7.4f} ({row['Strength']})")
        else:
            print("❌ No valid correlations found!")
            print("\n🔍 TROUBLESHOOTING:")
            print("1. Check if the data file 'B题附件：data.csv' contains market factor columns")
            print("2. Expected columns include: SP_close, DJ_close, EU_Price, OF_Price, etc.")
            print("3. Verify column names match the expected format")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print("📂 Folder structure:")
        print("   ├── README.md                          # Analysis overview")  
        if len(correlation_df) > 0:
            print("   ├── plot/                             # 6 correlation visualizations")
            print("   │   ├── 01_correlation_heatmap.png")
            print("   │   ├── 02_correlation_barplot.png") 
            print("   │   ├── 03_scatter_plots_top6.png")
            print("   │   ├── 04_normalized_timeseries.png")
            print("   │   ├── 05_correlation_summary_stats.png")
            print("   │   └── 06_rolling_correlations.png")
        print("   ├── report/                           # Analysis reports")
        print("   │   └── correlation_analysis_report.txt")
        print("   └── data/                             # Correlation data")
        if len(correlation_df) > 0:
            print("       ├── correlation_matrix.csv")
        print("       └── correlation_results.csv")
        
        if len(correlation_df) > 0:
            print("\n✅ Question 2 correlation analysis completed successfully!")
        else:
            print("\n⚠️  Question 2 completed with warnings - no valid correlations found")
            print("Please check the data file and column names for troubleshooting.")
        
    except Exception as e:
        print(f"❌ Error occurred during correlation analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()