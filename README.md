# 基于智能手机感知数据的心理压力评估

本项目是对论文《基于智能手机感知数据的心理压力评估方法》的复现实现。该方法使用智能手机传感器数据（包括WiFi、活动、对话、音频、蓝牙和屏幕使用数据）来评估用户的心理压力状态。

## 项目结构

```
.
├── README.md
├── requirements.txt
└── src
    ├── feature_extraction.py  # 特征提取模块
    ├── model.py              # 模型训练和评估模块
    └── main.py              # 主程序
```

## 数据集要求

本项目使用StudentLife数据集，数据集应按以下结构组织：

```
data/dataset/
├── sensing/
│   ├── wifi_location/
│   ├── activity/
│   ├── conversation/
│   ├── audio/
│   ├── bluetooth/
│   └── screen/
├── survey/
│   └── stress/
└── metadata/
    └── poi_locations.csv
```

## 安装依赖

1. 创建并激活Python虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据集：
   - 将StudentLife数据集按上述结构放置在`data/dataset`目录下

2. 运行程序：
```bash
python src/main.py
```

程序将：
- 加载并预处理数据
- 提取特征
- 训练模型
- 输出模型评估结果

## 特征说明

本项目提取的特征包括：

1. POI相关特征：
   - 不同地点的停留时间（教学区、住宿区、餐饮健康区等）
   - 地点分布熵

2. 活动相关特征：
   - 不同活动状态的时长（静止、走路、跑步等）
   - 活动状态分布熵

3. 社交相关特征：
   - 对话次数和时长
   - 不同地点的对话情况
   - 环境音频类型分布

4. 蓝牙相关特征：
   - 周围设备数量
   - 不同地点的设备数量分布

5. 睡眠相关特征：
   - 睡眠时间
   - 睡眠时长

## 模型说明

本项目使用基于随机森林的协同训练方法：

1. 特征选择：使用随机森林的特征重要性进行特征选择

2. 协同训练：
   - 使用有标签数据训练初始模型
   - 用初始模型为无标签数据生成伪标签
   - 迭代训练两个模型，相互学习

3. 集成预测：
   - 结合两个模型的预测结果
   - 使用加权投票方式得到最终预测

## 参考文献

[1] 王丰等. 基于智能手机感知数据的心理压力评估方法[J]. 计算机研究与发展. 