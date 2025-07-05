import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import joblib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# 机器学习导入
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, GridSearchCV, 
    RandomizedSearchCV, cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, RFE

# 统计分析导入
from scipy import stats
from scipy.stats import jarque_bera, shapiro

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_prediction_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置中文字体和图表样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')


@dataclass
class EnhancedConfig:
    """增强配置类"""
    # 数据参数
    data_file: str = 'B题附件：data.csv'
    target_column: str = 'Close'
    date_column: str = 'Date'
    
    # 特征工程参数
    ma_periods: List[int] = None
    lag_periods: List[int] = None
    rolling_windows: List[int] = None
    
    # 模型参数
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1
    
    # 输出参数
    output_dir: str = 'results/Q3_enhanced'
    save_models: bool = True
    create_visualizations: bool = True
    
    # 高级参数
    enable_hyperparameter_tuning: bool = True
    enable_feature_selection: bool = True
    n_features_to_select: int = 30
    
    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 200]
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 5, 10, 20]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 60]


class EnhancedGoldPricePredictor:
    """增强版黄金价格预测模型"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None
        self.feature_names: List[str] = []
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.scaler: Optional[Any] = None
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ['plots', 'reports', 'data', 'models']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """加载和准备数据集"""
        logger.info("开始加载和准备数据...")
        
        try:
            # 加载数据
            df = pd.read_csv(self.config.data_file)
            logger.info(f"数据加载完成: {df.shape[0]} 行, {df.shape[1]} 列")
            
            # 增强日期解析
            df[self.config.date_column] = self._parse_dates(df[self.config.date_column])
            
            # 移除日期无效或目标值缺失的行
            initial_size = len(df)
            df = df.dropna(subset=[self.config.date_column, self.config.target_column])
            logger.info(f"移除了 {initial_size - len(df)} 行缺失数据")
            
            # 按日期排序
            df = df.sort_values(self.config.date_column).reset_index(drop=True)
            
            # 数据质量检查
            self._data_quality_checks(df)
            
            self.data = df
            logger.info(f"数据准备完成: {len(df)} 行, 日期范围: {df[self.config.date_column].min()} 到 {df[self.config.date_column].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"数据加载错误: {e}")
            raise
    
    def _parse_dates(self, date_series: pd.Series) -> pd.Series:
        """增强日期解析"""
        # 尝试多种日期格式
        for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
            try:
                return pd.to_datetime(date_series, format=fmt)
            except:
                continue
        
        # 回退到自动解析
        return pd.to_datetime(date_series, infer_datetime_format=True, errors='coerce')
    
    def _data_quality_checks(self, df: pd.DataFrame) -> None:
        """执行全面的数据质量检查"""
        logger.info("执行数据质量检查...")
        
        # 检查重复项
        duplicates = df.duplicated(subset=[self.config.date_column]).sum()
        if duplicates > 0:
            logger.warning(f"发现 {duplicates} 个重复日期")
        
        # 检查目标变量的异常值
        target_data = df[self.config.target_column]
        q1, q3 = target_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((target_data < (q1 - 1.5 * iqr)) | (target_data > (q3 + 1.5 * iqr))).sum()
        logger.info(f"目标变量中发现 {outliers} 个潜在异常值")
        
        # 检查数据分布
        skewness = stats.skew(target_data.dropna())
        kurtosis = stats.kurtosis(target_data.dropna())
        logger.info(f"目标变量 - 偏度: {skewness:.3f}, 峰度: {kurtosis:.3f}")
    
    def create_advanced_features(self) -> pd.DataFrame:
        """创建全面的特征集"""
        logger.info("创建高级特征...")
        
        df = self.data.copy()
        
        # 基础价格特征
        df = self._create_price_features(df)
        
        # 技术指标
        df = self._create_technical_indicators(df)
        
        # 市场因子特征（基于Q2相关性分析）
        df = self._create_market_features(df)
        
        # 时间特征
        df = self._create_time_features(df)
        
        # 统计特征
        df = self._create_statistical_features(df)
        
        # 滞后和滚动特征
        df = self._create_lag_rolling_features(df)
        
        # 移除包含NaN的行
        df_clean = df.dropna()
        logger.info(f"特征创建完成。清洁数据集: {df_clean.shape}")
        
        # 分离特征和目标
        feature_cols = [col for col in df_clean.columns 
                       if col not in [self.config.target_column, self.config.date_column]]
        
        self.features = df_clean[feature_cols]
        self.target = df_clean[self.config.target_column]
        self.feature_names = feature_cols
        
        logger.info(f"总特征数: {len(feature_cols)}")
        
        return df_clean
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基础价格特征"""
        target = self.config.target_column
        
        # 价格变化
        for period in [1, 5, 20]:
            df[f'price_change_{period}d'] = df[target].diff(period)
            df[f'price_pct_change_{period}d'] = df[target].pct_change(period)
        
        # 价格比率
        if 'High' in df.columns and 'Low' in df.columns:
            df['high_low_ratio'] = df['High'] / df['Low']
        if 'Open' in df.columns:
            df['close_open_ratio'] = df[target] / df['Open']
        
        # 价格位置指标
        for period in [10, 20, 50]:
            df[f'price_position_{period}d'] = (
                (df[target] - df[target].rolling(period).min()) / 
                (df[target].rolling(period).max() - df[target].rolling(period).min())
            )
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标"""
        target = self.config.target_column
        
        # 移动平均线
        for period in self.config.ma_periods:
            df[f'sma_{period}'] = df[target].rolling(window=period).mean()
            df[f'ema_{period}'] = df[target].ewm(span=period).mean()
            
            # 价格与移动平均线的关系
            df[f'price_sma_ratio_{period}'] = df[target] / df[f'sma_{period}']
            df[f'price_ema_ratio_{period}'] = df[target] / df[f'ema_{period}']
        
        # RSI
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi_14'] = calculate_rsi(df[target])
        df['rsi_7'] = calculate_rsi(df[target], 7)
        df['rsi_21'] = calculate_rsi(df[target], 21)
        
        # MACD
        ema_12 = df[target].ewm(span=12).mean()
        ema_26 = df[target].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 布林带
        for period in [10, 20]:
            sma = df[target].rolling(window=period).mean()
            std = df[target].rolling(window=period).std()
            df[f'bollinger_upper_{period}'] = sma + (std * 2)
            df[f'bollinger_lower_{period}'] = sma - (std * 2)
            df[f'bollinger_width_{period}'] = df[f'bollinger_upper_{period}'] - df[f'bollinger_lower_{period}']
            df[f'bollinger_position_{period}'] = (df[target] - df[f'bollinger_lower_{period}']) / df[f'bollinger_width_{period}']
        
        # 随机指标
        if 'High' in df.columns and 'Low' in df.columns:
            for period in [14, 21]:
                lowest_low = df['Low'].rolling(window=period).min()
                highest_high = df['High'].rolling(window=period).max()
                df[f'stoch_k_{period}'] = 100 * (df[target] - lowest_low) / (highest_high - lowest_low)
                df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        
        return df
    
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建市场因子特征（基于Q2相关性分析）"""
        market_factors = [
            'GDX_Close',    # 黄金矿业ETF (r=0.9755)
            'PLT_Price',    # 铂金价格 (r=0.7759)
            'USDI_Price',   # 美元指数 (r=-0.7216)
            'OF_Price',     # 原油价格 (r=0.7107)
            'SP_close',     # 标普500 (r=-0.6843)
            'USO_Close',    # 原油ETF (r=0.6357)
            'OS_Price',     # 白银价格 (r=0.6308)
        ]
        
        for factor in market_factors:
            if factor in df.columns:
                # 与黄金的价格比率
                df[f'{factor}_gold_ratio'] = df[factor] / df[self.config.target_column]
                
                # 相对变化
                df[f'{factor}_pct_change_1d'] = df[factor].pct_change()
                df[f'{factor}_pct_change_5d'] = df[factor].pct_change(5)
                
                # 滚动相关性
                for window in [20, 60]:
                    df[f'{factor}_correlation_{window}d'] = (
                        df[factor].rolling(window).corr(df[self.config.target_column])
                    )
                
                # 相对强度
                df[f'{factor}_relative_strength'] = (
                    df[factor] / df[factor].rolling(window=20).mean()
                )
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间特征"""
        df['year'] = df[self.config.date_column].dt.year
        df['month'] = df[self.config.date_column].dt.month
        df['day_of_week'] = df[self.config.date_column].dt.dayofweek
        df['day_of_year'] = df[self.config.date_column].dt.dayofyear
        df['quarter'] = df[self.config.date_column].dt.quarter
        df['week_of_year'] = df[self.config.date_column].dt.isocalendar().week
        
        # 周期性编码
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # 是否为月末/季末/年末
        df['is_month_end'] = df[self.config.date_column].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df[self.config.date_column].dt.is_quarter_end.astype(int)
        df['is_year_end'] = df[self.config.date_column].dt.is_year_end.astype(int)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建统计特征"""
        target = self.config.target_column
        
        for window in [10, 20, 60]:
            # 滚动统计
            df[f'rolling_mean_{window}'] = df[target].rolling(window).mean()
            df[f'rolling_std_{window}'] = df[target].rolling(window).std()
            df[f'rolling_var_{window}'] = df[target].rolling(window).var()
            df[f'rolling_skew_{window}'] = df[target].rolling(window).skew()
            df[f'rolling_kurt_{window}'] = df[target].rolling(window).kurt()
            
            # 滚动分位数
            df[f'rolling_q25_{window}'] = df[target].rolling(window).quantile(0.25)
            df[f'rolling_q75_{window}'] = df[target].rolling(window).quantile(0.75)
            df[f'rolling_median_{window}'] = df[target].rolling(window).median()
            
            # 与滚动统计的距离
            df[f'distance_from_mean_{window}'] = df[target] - df[f'rolling_mean_{window}']
            df[f'distance_from_mean_normalized_{window}'] = (
                df[f'distance_from_mean_{window}'] / df[f'rolling_std_{window}']
            )
            
            # 滚动范围
            df[f'rolling_range_{window}'] = (
                df[target].rolling(window).max() - df[target].rolling(window).min()
            )
        
        return df
    
    def _create_lag_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建滞后和滚动特征"""
        target = self.config.target_column
        
        # 滞后特征
        for lag in self.config.lag_periods:
            df[f'{target}_lag_{lag}'] = df[target].shift(lag)
            
            # 滞后变化
            if lag > 1:
                df[f'{target}_lag_change_{lag}'] = df[target] - df[f'{target}_lag_{lag}']
                df[f'{target}_lag_pct_change_{lag}'] = (
                    (df[target] - df[f'{target}_lag_{lag}']) / df[f'{target}_lag_{lag}']
                )
        
        # 滚动窗口特征
        for window in self.config.rolling_windows:
            df[f'{target}_rolling_mean_{window}'] = df[target].rolling(window).mean()
            df[f'{target}_rolling_std_{window}'] = df[target].rolling(window).std()
            df[f'{target}_rolling_min_{window}'] = df[target].rolling(window).min()
            df[f'{target}_rolling_max_{window}'] = df[target].rolling(window).max()
            
            # 滚动趋势
            def rolling_trend(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                if np.all(np.isnan(y)):
                    return np.nan
                try:
                    slope, _ = np.polyfit(x, y, 1)
                    return slope
                except:
                    return np.nan
            
            df[f'{target}_rolling_trend_{window}'] = (
                df[target].rolling(window).apply(rolling_trend, raw=False)
            )
        
        return df
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """特征选择"""
        if not self.config.enable_feature_selection:
            return X
        
        logger.info(f"执行特征选择，从 {X.shape[1]} 个特征中选择 {self.config.n_features_to_select} 个")
        
        # 使用SelectKBest进行特征选择
        selector = SelectKBest(score_func=f_regression, k=self.config.n_features_to_select)
        X_selected = selector.fit_transform(X, y)
        
        # 获取选中的特征名称
        selected_features = X.columns[selector.get_support()]
        logger.info(f"选中的特征: {list(selected_features)}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def create_enhanced_models(self) -> Dict[str, Pipeline]:
        """创建增强模型管道"""
        logger.info("创建增强模型...")
        
        models = {
            'linear_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            
            'ridge_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(random_state=self.config.random_state))
            ]),
            
            'lasso_regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(random_state=self.config.random_state, max_iter=2000))
            ]),
            
            'elastic_net': Pipeline([
                ('scaler', StandardScaler()),
                ('model', ElasticNet(random_state=self.config.random_state, max_iter=2000))
            ]),
            
            'random_forest': Pipeline([
                ('scaler', RobustScaler()),
                ('model', RandomForestRegressor(
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                ))
            ]),
            
            'extra_trees': Pipeline([
                ('scaler', RobustScaler()),
                ('model', ExtraTreesRegressor(
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                ))
            ]),
            
            'gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(random_state=self.config.random_state))
            ]),
            
            'svr_rbf': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='rbf'))
            ]),
            
            'svr_linear': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVR(kernel='linear'))
            ]),
            
            'mlp_regressor': Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPRegressor(
                    random_state=self.config.random_state,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.1
                ))
            ])
        }
        
        self.models = models
        return models
    
    def optimize_hyperparameters(self, model_name: str, param_grid: Dict) -> Any:
        """优化超参数"""
        if not self.config.enable_hyperparameter_tuning:
            return self.models[model_name]
        
        logger.info(f"为 {model_name} 优化超参数")
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        # 使用RandomizedSearchCV提高效率
        search = RandomizedSearchCV(
            self.models[model_name],
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_iter=10,  # 减少迭代次数以提高速度
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=0
        )
        
        search.fit(self.X_train, self.y_train)
        
        logger.info(f"{model_name} 最佳参数: {search.best_params_}")
        return search.best_estimator_
    
    def train_and_evaluate_models(self) -> Dict[str, Dict]:
        """训练和评估所有模型"""
        logger.info("训练和评估模型...")
        
        # 分割数据
        self._split_data()
        
        # 特征选择
        if self.config.enable_feature_selection:
            self.X_train = self.feature_selection(self.X_train, self.y_train)
            # 对测试集应用相同的特征选择
            self.X_test = self.X_test[self.X_train.columns]
        
        # 定义超参数网格
        param_grids = {
            'random_forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.05, 0.1, 0.2],
                'model__max_depth': [3, 6, 9]
            },
            'svr_rbf': {
                'model__C': [0.1, 1, 10],
                'model__gamma': ['scale', 'auto']
            },
            'ridge_regression': {
                'model__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso_regression': {
                'model__alpha': [0.01, 0.1, 1.0, 10.0]
            }
        }
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"训练 {name}...")
            
            try:
                # 如果有参数网格，进行超参数优化
                if name in param_grids and self.config.enable_hyperparameter_tuning:
                    model = self.optimize_hyperparameters(name, param_grids[name])
                
                # 训练模型
                model.fit(self.X_train, self.y_train)
                
                # 预测
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                
                # 计算指标
                metrics = self._calculate_comprehensive_metrics(
                    self.y_train, y_train_pred, self.y_test, y_test_pred
                )
                
                # 存储结果
                results[name] = {
                    'model': model,
                    'train_predictions': y_train_pred,
                    'test_predictions': y_test_pred,
                    **metrics
                }
                
                logger.info(f"{name} - 测试 R²: {metrics['test_r2']:.4f}, 测试 RMSE: {metrics['test_rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"训练 {name} 时出错: {e}")
                continue
        
        self.results = results
        
        # 找到最佳模型
        if results:
            self.best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
            logger.info(f"最佳模型: {self.best_model_name}")
        
        return results
    
    def _split_data(self) -> None:
        """使用时间序列方法分割数据"""
        # 时间序列分割
        split_idx = int(len(self.features) * (1 - self.config.test_size))
        
        self.X_train = self.features.iloc[:split_idx].copy()
        self.X_test = self.features.iloc[split_idx:].copy()
        self.y_train = self.target.iloc[:split_idx].copy()
        self.y_test = self.target.iloc[split_idx:].copy()
        
        logger.info(f"数据分割 - 训练: {len(self.X_train)}, 测试: {len(self.X_test)}")
    
    def _calculate_comprehensive_metrics(self, y_train_true, y_train_pred, y_test_true, y_test_pred) -> Dict:
        """计算全面的评估指标"""
        
        def safe_mape(y_true, y_pred):
            """安全计算MAPE，避免除零错误"""
            mask = y_true != 0
            if mask.sum() == 0:
                return np.inf
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        metrics = {
            # 训练指标
            'train_mse': mean_squared_error(y_train_true, y_train_pred),
            'train_mae': mean_absolute_error(y_train_true, y_train_pred),
            'train_r2': r2_score(y_train_true, y_train_pred),
            'train_mape': safe_mape(y_train_true, y_train_pred),
            
            # 测试指标
            'test_mse': mean_squared_error(y_test_true, y_test_pred),
            'test_mae': mean_absolute_error(y_test_true, y_test_pred),
            'test_r2': r2_score(y_test_true, y_test_pred),
            'test_mape': safe_mape(y_test_true, y_test_pred),
        }
        
        # 衍生指标
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
        metrics['overfitting'] = metrics['train_r2'] - metrics['test_r2']
        
        return metrics
    
    def create_comprehensive_visualizations(self) -> None:
        """创建全面的可视化"""
        if not self.config.create_visualizations:
            return
            
        logger.info("创建综合可视化...")
        
        # 设置样式
        sns.set_palette("husl")
        
        # 1. 模型性能比较
        self._plot_model_performance_comparison()
        
        # 2. 最佳模型分析
        self._plot_best_model_analysis()
        
        # 3. 特征重要性
        self._plot_feature_importance()
        
        # 4. 残差分析
        self._plot_residual_analysis()
        
        # 5. 学习曲线
        self._plot_learning_curves()
        
        # 6. 时间序列预测
        self._plot_time_series_prediction()
        
        logger.info(f"可视化已保存到 {self.output_dir / 'plots'}")
    
    def _plot_model_performance_comparison(self) -> None:
        """绘制模型性能比较"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')

        models = list(self.results.keys())
        test_r2 = [self.results[m]['test_r2'] for m in models]
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        test_mape = [self.results[m]['test_mape'] for m in models]
        overfitting = [self.results[m]['overfitting'] for m in models]

        # R²比较
        axes[0, 0].bar(models, test_r2, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Test R² Score')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # RMSE比较
        axes[0, 1].bar(models, test_rmse, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Test RMSE')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # MAPE比较
        axes[0, 2].bar(models, test_mape, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title('Test MAPE (%)')
        axes[0, 2].set_ylabel('MAPE (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)

        # 过拟合分析
        colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in overfitting]
        axes[1, 0].bar(models, overfitting, alpha=0.7, color=colors)
        axes[1, 0].set_title('Overfitting Analysis')
        axes[1, 0].set_ylabel('Train R² - Test R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 模型排名
        ranking_data = pd.DataFrame({
            'Model': models,
            'Test_R2': test_r2
        }).sort_values('Test_R2', ascending=True)

        axes[1, 1].barh(ranking_data['Model'], ranking_data['Test_R2'], alpha=0.7, color='purple')
        axes[1, 1].set_title('Model Ranking (by R²)')
        axes[1, 1].set_xlabel('R² Score')
        axes[1, 1].grid(True, alpha=0.3)

        # 性能总结
        best_model = self.results[self.best_model_name]
        summary_text = f"""Best Model: {self.best_model_name}
Test R²: {best_model['test_r2']:.4f}
Test RMSE: ${best_model['test_rmse']:.2f}
Test MAE: ${best_model['test_mae']:.2f}
Test MAPE: {best_model['test_mape']:.2f}%
Overfitting: {best_model['overfitting']:.4f}"""

        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_title('Best Model Summary')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / '01_model_performance_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_best_model_analysis(self) -> None:
        """绘制最佳模型详细分析"""
        best_model = self.results[self.best_model_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Best Model Detailed Analysis: {self.best_model_name}', fontsize=16, fontweight='bold')
        
        # 训练集：预测 vs 实际
        axes[0, 0].scatter(self.y_train, best_model['train_predictions'], alpha=0.6, s=20)
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], 
                       [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Predicted Price')
        axes[0, 0].set_title(f'Train Set (R² = {best_model["train_r2"]:.4f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 测试集：预测 vs 实际
        axes[0, 1].scatter(self.y_test, best_model['test_predictions'], alpha=0.6, s=20, color='orange')
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].set_title(f'Test Set (R² = {best_model["test_r2"]:.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 残差图
        test_residuals = self.y_test - best_model['test_predictions']
        axes[0, 2].scatter(best_model['test_predictions'], test_residuals, alpha=0.6)
        axes[0, 2].axhline(y=0, color='red', linestyle='--')
        axes[0, 2].set_xlabel('Predicted Price')
        axes[0, 2].set_ylabel('Residual')
        axes[0, 2].set_title('Residual Analysis')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 残差分布
        axes[1, 0].hist(test_residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(test_residuals.mean(), color='red', linestyle='--', 
                          label=f'均值: {test_residuals.mean():.4f}')
        axes[1, 0].axvline(test_residuals.std(), color='orange', linestyle='--', 
                          label=f'标准差: {test_residuals.std():.4f}')
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q图
        stats.probplot(test_residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Residual Normality Test (Q-Q Plot)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 累积误差
        cumulative_error = np.cumsum(np.abs(test_residuals))
        axes[1, 2].plot(range(len(cumulative_error)), cumulative_error, linewidth=2)
        axes[1, 2].set_xlabel('Test Sample Index')
        axes[1, 2].set_ylabel('Cumulative Absolute Error')
        axes[1, 2].set_title('Cumulative Prediction Error')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / '02_best_model_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self) -> None:
        """绘制特征重要性"""
        best_model = self.results[self.best_model_name]['model']
        
        # 检查模型是否有特征重要性
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            importance = best_model.named_steps['model'].feature_importances_
            
            # 创建重要性DataFrame
            feature_names = self.X_train.columns if hasattr(self, 'X_train') else self.feature_names
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(20)  # 前20个特征
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue', alpha=0.7)
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importances: {self.best_model_name}')
            plt.gca().invert_yaxis()
            
            # 添加数值标签
            for i, v in enumerate(importance_df['importance']):
                plt.text(v + 0.001, i, f'{v:.3f}', va='center')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / '03_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.info(f"模型 {self.best_model_name} 不支持特征重要性分析")
    
    def _plot_residual_analysis(self) -> None:
        """高级残差分析"""
        best_model = self.results[self.best_model_name]
        residuals = self.y_test - best_model['test_predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced Residual Analysis', fontsize=16, fontweight='bold')
        
        # 残差分布与正态分布对比
        axes[0, 0].hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
                       label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        axes[0, 0].set_title('Residual Distribution')
        axes[0, 0].set_xlabel('Residual')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 标准化残差
        standardized_residuals = residuals / np.std(residuals)
        axes[0, 1].scatter(best_model['test_predictions'], np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
        axes[0, 1].set_xlabel('Fitted Value')
        axes[0, 1].set_ylabel('√|Standardized Residual|')
        axes[0, 1].set_title('Scale-Location Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 残差随时间变化
        axes[1, 0].plot(range(len(residuals)), residuals, alpha=0.7, linewidth=1)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].set_title('Residuals Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 残差自相关（简化版）
        lags = range(1, min(21, len(residuals)//4))
        autocorr = [np.corrcoef(residuals[:-lag], residuals[lag:])[0,1] for lag in lags]
        axes[1, 1].bar(lags, autocorr, alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='-')
        axes[1, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].set_title('Residual Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / '04_residual_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_curves(self) -> None:
        """绘制学习曲线"""
        best_model = self.results[self.best_model_name]['model']
        
        train_sizes, train_scores, val_scores = learning_curve(
            best_model, self.X_train, self.y_train,
            cv=TimeSeriesSplit(n_splits=3),
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2',
            n_jobs=self.config.n_jobs
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Train Score', color='blue')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score', color='red')
        plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color='blue')
        plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                        np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('R² Score')
        plt.title(f'Learning Curve: {self.best_model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / '05_learning_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series_prediction(self) -> None:
        """绘制时间序列预测"""
        best_model = self.results[self.best_model_name]

        # 获取日期数据
        if hasattr(self, 'data') and self.config.date_column in self.data.columns:
            dates = self.data[self.config.date_column].iloc[-len(self.y_test):]
        else:
            dates = pd.date_range(start='2020-01-01', periods=len(self.y_test), freq='D')

        plt.figure(figsize=(15, 8))

        # 绘制实际值和预测值
        plt.plot(dates, self.y_test.values, label='Actual Price', color='blue', linewidth=2)
        plt.plot(dates, best_model['test_predictions'], label='Predicted Price',
                 color='red', linewidth=2, alpha=0.8)

        # 添加置信区间（基于RMSE）
        rmse = best_model['test_rmse']
        plt.fill_between(dates,
                         best_model['test_predictions'] - rmse,
                         best_model['test_predictions'] + rmse,
                         alpha=0.2, color='red', label=f'±{rmse:.2f} Confidence Interval')

        plt.xlabel('Date')
        plt.ylabel('Gold Price (USD)')
        plt.title(f'Time Series Prediction: {self.best_model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / '06_time_series_prediction.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_results(self) -> None:
        """保存综合结果"""
        logger.info("保存综合结果...")
        
        # 保存模型性能
        performance_df = pd.DataFrame([
            {
                'model': name,
                'train_r2': result['train_r2'],
                'test_r2': result['test_r2'],
                'train_rmse': result['train_rmse'],
                'test_rmse': result['test_rmse'],
                'train_mae': result['train_mae'],
                'test_mae': result['test_mae'],
                'train_mape': result['train_mape'],
                'test_mape': result['test_mape'],
                'overfitting': result['overfitting']
            }
            for name, result in self.results.items()
        ]).sort_values('test_r2', ascending=False)
        
        performance_df.to_csv(self.output_dir / 'data' / 'model_performance.csv', index=False)
        
        # 保存预测结果
        best_model = self.results[self.best_model_name]
        predictions_df = pd.DataFrame({
            'actual': self.y_test.values,
            'predicted': best_model['test_predictions'],
            'residual': self.y_test.values - best_model['test_predictions'],
            'absolute_error': np.abs(self.y_test.values - best_model['test_predictions']),
            'percentage_error': np.abs((self.y_test.values - best_model['test_predictions']) / self.y_test.values * 100)
        })
        predictions_df.to_csv(self.output_dir / 'data' / 'predictions.csv', index=False)
        
        # 保存特征重要性（如果可用）
        if (hasattr(self.results[self.best_model_name]['model'].named_steps['model'], 'feature_importances_')):
            feature_names = self.X_train.columns if hasattr(self, 'X_train') else self.feature_names
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.results[self.best_model_name]['model'].named_steps['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            importance_df.to_csv(self.output_dir / 'data' / 'feature_importance.csv', index=False)
        
        # 保存模型
        if self.config.save_models:
            for name, result in self.results.items():
                joblib.dump(result['model'], self.output_dir / 'models' / f'{name}.pkl')
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {self.output_dir}")
    
    def generate_comprehensive_report(self) -> None:
        """生成综合分析报告"""
        logger.info("生成综合分析报告...")
        
        best_model = self.results[self.best_model_name]
        
        # 计算统计检验
        residuals = self.y_test - best_model['test_predictions']
        try:
            jb_stat, jb_p = jarque_bera(residuals)
            sw_stat, sw_p = shapiro(residuals[:min(5000, len(residuals))])
        except:
            jb_stat, jb_p = np.nan, np.nan
            sw_stat, sw_p = np.nan, np.nan
        
        report = f"""# 增强版黄金价格预测模型分析报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 执行摘要
- **数据集**: {len(self.data)} 个观测值
- **特征**: {len(self.feature_names)} 个工程特征
- **最佳模型**: {self.best_model_name}
- **测试 R²**: {best_model['test_r2']:.4f}
- **测试 RMSE**: ${best_model['test_rmse']:.2f}
- **测试 MAPE**: {best_model['test_mape']:.2f}%

## 模型性能总结

"""
        
        # 添加性能表格
        performance_df = pd.DataFrame([
            {
                'Model': name,
                'Test R²': f"{result['test_r2']:.4f}",
                'Test RMSE': f"${result['test_rmse']:.2f}",
                'Test MAPE': f"{result['test_mape']:.2f}%",
                'Overfitting': f"{result['overfitting']:.4f}"
            }
            for name, result in self.results.items()
        ])
        
        report += performance_df.sort_values('Test R²', ascending=False).to_string(index=False)
        
        report += f"""

## 最佳模型分析: {self.best_model_name}
- **预测准确度**: {(1 - best_model['test_mape']/100)*100:.1f}% 平均预测准确度
- **误差范围**: ±${best_model['test_rmse']:.2f} (1个标准差)
- **过拟合程度**: {best_model['overfitting']:.4f} ({'低' if best_model['overfitting'] < 0.05 else '中等' if best_model['overfitting'] < 0.1 else '高'})

## 统计检验结果
- **Jarque-Bera检验**: 统计量={jb_stat:.4f}, p值={jb_p:.4f}
- **Shapiro-Wilk检验**: 统计量={sw_stat:.4f}, p值={sw_p:.4f}
- **残差正态性**: {'正态分布' if sw_p > 0.05 else '非正态分布'} 

## 商业应用建议

### 投资策略
- **预测期限**: 适合{'短期' if best_model['test_r2'] > 0.9 else '中期'}价格预测
- **风险管理**: 预期预测误差在 ${best_model['test_rmse']:.2f} 范围内
- **交易信号**: 模型置信度{'高' if best_model['test_r2'] > 0.9 else '中等' if best_model['test_r2'] > 0.8 else '低'}

### 风险评估
- **预测区间**: 68%置信区间为 ±${best_model['test_rmse']:.2f}
- **最大误差**: {np.max(np.abs(residuals)):.2f}
- **平均误差**: ${best_model['test_mae']:.2f}

### 模型监控建议
1. **部署建议**: {'推荐部署' if best_model['test_r2'] > 0.85 else '需要进一步改进'}
2. **监控频率**: 监控模型性能退化
3. **重训练**: {'月度' if best_model['test_r2'] > 0.9 else '周度'}模型更新建议
4. **风险控制**: 实施预测置信区间

## 技术细节

### 特征工程
- 移动平均: {len(self.config.ma_periods)} 个周期
- 滞后特征: {len(self.config.lag_periods)} 个滞后期
- 技术指标: RSI, MACD, 布林带等
- 市场因子: 基于Q2相关性分析的7个关键因子

### 模型配置
- 交叉验证: {self.config.cv_folds}折时间序列交叉验证
- 超参数优化: {'启用' if self.config.enable_hyperparameter_tuning else '禁用'}
- 特征选择: {'启用' if self.config.enable_feature_selection else '禁用'}

### 生成文件
- model_performance.csv: 详细模型比较
- predictions.csv: 测试集预测和误差
- feature_importance.csv: 特征重要性排名
- 模型图表: 综合可视化分析
- 训练模型: 序列化模型对象

---
*由增强版黄金价格预测系统生成*
"""
        
        # 保存报告
        with open(self.output_dir / 'reports' / 'comprehensive_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已生成: {self.output_dir / 'reports' / 'comprehensive_analysis_report.md'}")


def main():
    """主执行函数"""
    logger.info("启动增强版黄金价格预测分析")
    print("=" * 80)
    print("增强版黄金价格预测模型 - 问题3")
    print("=" * 80)
    
    # 创建配置
    config = EnhancedConfig()
    
    # 初始化预测器
    predictor = EnhancedGoldPricePredictor(config)
    
    try:
        # 执行分析流程
        print("\n1. 加载和准备数据...")
        predictor.load_and_prepare_data()
        
        print("2. 创建高级特征...")
        predictor.create_advanced_features()
        
        print("3. 创建增强模型...")
        predictor.create_enhanced_models()
        
        print("4. 训练和评估模型...")
        predictor.train_and_evaluate_models()
        
        print("5. 创建可视化...")
        predictor.create_comprehensive_visualizations()
        
        print("6. 保存结果...")
        predictor.save_comprehensive_results()
        
        print("7. 生成报告...")
        predictor.generate_comprehensive_report()
        
        # 打印最终结果
        best_model = predictor.results[predictor.best_model_name]
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        print(f"📊 训练模型数量: {len(predictor.results)}")
        print(f"🏆 最佳模型: {predictor.best_model_name}")
        print(f"📈 测试 R²: {best_model['test_r2']:.4f}")
        print(f"💰 测试 RMSE: ${best_model['test_rmse']:.2f}")
        print(f"📉 测试 MAE: ${best_model['test_mae']:.2f}")
        print(f"🎯 测试 MAPE: {best_model['test_mape']:.2f}%")
        print(f"⚖️  过拟合程度: {best_model['overfitting']:.4f}")
        
        accuracy = (1 - best_model['test_mape']/100) * 100
        print(f"✅ 平均预测准确度: {accuracy:.1f}%")
        
        print(f"\n📁 结果保存至: {config.output_dir}")
        print("\n📂 生成的文件:")
        print("   ├── plots/                    # 6个高质量可视化图表")
        print("   ├── reports/                  # 综合分析报告")
        print("   ├── data/                     # 性能数据和预测结果")
        print("   └── models/                   # 训练好的模型文件")
        
        print("\n🎉 问题3增强版预测建模完成！")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"分析流程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()