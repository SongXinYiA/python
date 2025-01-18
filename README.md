# 运动检测系统

该项目是一个多功能运动检测系统，支持俯卧撑、跳绳、蹲起、开合跳等运动的检测和评分。利用 **MediaPipe** 库进行姿势估计，并实时分析运动质量，给予评分。

## 项目功能

### 1. 运动检测
- 实时检测用户进行的运动（如俯卧撑、跳绳、蹲起等）。
- 系统根据检测到的姿势进行分析，评估运动的质量，并给出评分。
- 根据运动类型进行不同的标准化评分。

### 2. 评分系统
- 基于运动的深度、速度、稳定性等多个维度，自动评分。
- 系统会根据评分标准给予用户反馈，帮助用户调整姿势。
  
### 3. 视频流显示
- 用户的运动视频会被实时显示，帮助调整姿势。

## 依赖

- **Flask**: Web框架，用于处理请求和响应。
- **OpenCV**: 用于视频流的处理和显示。
- **MediaPipe**: 用于姿态估计，获取用户的运动数据。

## 功能介绍

### 首页:
- 用户可以选择不同的运动检测类型：俯卧撑检测、跳绳检测、蹲起检测、开合跳检测。

### 运动检测:
- 系统会通过摄像头实时捕捉用户的运动数据，并进行分析。
- 根据不同的运动类型，系统会分析用户的运动深度、速度、稳定性等指标，并给出评分。

### 评分反馈:
- 每项运动有一个评分标准，系统会根据用户的运动表现给出一个分数。

## 注意事项
- 请确保在运行项目时，摄像头已正确连接并能正常工作。
- 系统的精确度取决于用户的姿势和摄像头角度。

## 运行应用

```bash
python app.py
