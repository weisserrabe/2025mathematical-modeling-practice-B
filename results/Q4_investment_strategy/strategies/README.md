# 投资策略配置文件说明

## 文件结构

### 1. base_strategy_config.json
基础策略配置，包含核心参数设置

### 2. portfolio_strategies.json  
投资组合分配策略，包含四种不同的资产配置方案

### 3. trading_signals_strategy.json
交易信号生成策略，基于多因子模型

### 4. risk_management_strategy.json
风险管理框架配置

### 5. strategy_templates.json
三种风险偏好的策略模板：保守型、稳健型、激进型

### 6. market_regime_strategies.json
不同市场环境下的策略调整方案

## 使用方法

1. 根据投资者风险偏好选择合适的策略模板
2. 基于当前市场状态调整策略参数
3. 定期根据回测结果优化配置
4. 严格执行风险管理规则

## 参数调整指南

- **保守型投资者**: 使用conservative模板，降低仓位和风险暴露
- **稳健型投资者**: 使用moderate模板，平衡收益和风险
- **激进型投资者**: 使用aggressive模板，追求更高收益

## 注意事项

- 所有策略配置基于历史数据分析
- 实际投资需考虑市场变化和个人情况
- 建议定期回测和调整策略参数
