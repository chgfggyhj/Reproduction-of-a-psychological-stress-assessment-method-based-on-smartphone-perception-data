import pandas as pd
import numpy as np
from pathlib import Path
from feature_extraction import FeatureExtractor
from model import StressPredictor
from sklearn.model_selection import train_test_split
import json

def load_stress_data(stress_path):
    """
    从JSON文件加载压力水平数据
    参数:
        stress_path: 包含压力数据文件的目录路径
    返回:
        包含压力水平记录的DataFrame
    """
    stress_data = []
    stress_files = list(stress_path.glob('*.json'))
    
    for f in stress_files:
        try:
            # 从文件名中提取用户ID
            user_id = str(f.stem).split('_')[1]
            with open(f, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            # 提取时间戳和压力水平
                            timestamp = entry.get('resp_time', 0)
                            level = entry.get('level')
                            
                            if timestamp and level is not None:
                                try:
                                    stress_level = int(level)
                                    stress_data.append({
                                        'timestamp': timestamp,
                                        'stress_level': stress_level,
                                        'user_id': user_id
                                    })
                                except (ValueError, TypeError):
                                    continue
        except Exception as e:
            print(f"警告: 加载压力文件 {f} 时出错: {e}")
    
    if stress_data:
        df = pd.DataFrame(stress_data)
        return df
    else:
        print("警告: 未找到有效的压力数据")
        return pd.DataFrame(columns=['timestamp', 'stress_level', 'user_id'])

def extract_location(loc_str):
    """
    从WiFi位置字符串中提取位置信息
    参数:
        loc_str: WiFi数据中的原始位置字符串
    返回:
        提取的位置名称
    """
    
    try:
        # 从不同格式模式中提取位置
        if 'in[' in loc_str:
            loc = loc_str.split('in[')[1].rstrip(']')
        elif 'near[' in loc_str:
            loc = loc_str.split('near[')[1].rstrip(']')
        else:
            return 'unknown'
        
        # 返回位置字符串的第一部分
        return loc.split(';')[0].strip()
    except Exception:
        return 'unknown'

def load_data(data_dir):
    """
    从各种CSV文件加载所有传感器数据
    参数:
        data_dir: 包含所有数据文件的根目录
    返回:
        包含每种传感器类型DataFrame的字典
    """
    data = {}
    
    # 加载WiFi位置数据
    print("正在加载WiFi数据...")
    wifi_path = Path(data_dir) / 'sensing/wifi_location'
    wifi_files = list(wifi_path.glob('*.csv'))
    wifi_data = []
    for f in wifi_files:
        try:
            user_id = str(f.stem).split('_')[-1]
            # 尝试自动推断分隔符，并跳过格式错误的行
            df = pd.read_csv(f, names=['timestamp', 'location'], skiprows=1, usecols=[0, 1])
            if 'location' in df.columns:
                df['wifi_ap'] = df['location'].apply(extract_location)
                df['user_id'] = user_id
                wifi_data.append(df)
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if wifi_data:
        data['wifi'] = pd.concat(wifi_data, ignore_index=True)
    else:
        print("警告: 未找到有效的WiFi数据")
        data['wifi'] = pd.DataFrame(columns=['timestamp', 'wifi_ap', 'user_id'])
    
    # 加载活动数据
    print("正在加载活动数据...")
    activity_path = Path(data_dir) / 'sensing/activity'
    activity_files = list(activity_path.glob('*.csv'))
    activity_data = []
    for f in activity_files:
        try:
            user_id = str(f.stem).split('_')[-1]
            chunks = pd.read_csv(f, chunksize=10000)
            df = pd.concat(chunks)

            if ' activity inference' in df.columns:
                df=df.rename(columns={' activity inference':'activity'})
            if 'activity' not in df.columns:
                print(f"警告: 文件 {f} 缺少 activity 列")
                continue
            df['activity'] = pd.to_numeric(df['activity'], errors='coerce')
            # 将数字活动类型映射为字符串
            activity_map = {0: 'stationary', 1: 'walking', 2: 'running', 3: 'unknown'}
            df['activity'] = df['activity'].map(activity_map).fillna('unknown')
            if 'timestamp' in df.columns:
                df['user_id'] = user_id
                activity_data.append(df)
            else:
                print(f"警告: 文件 {f} 缺少 timestamp 列")
                continue
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if activity_data:
        data['activity'] = pd.concat(activity_data, ignore_index=True)
    else:
        print("警告: 未找到有效的活动数据")
        data['activity'] = pd.DataFrame(columns=['timestamp', 'activity inference', 'user_id'])
    
    # 加载对话数据
    print("正在加载对话数据...")
    conversation_path = Path(data_dir) / 'sensing/conversation'
    conversation_files = list(conversation_path.glob('*.csv'))
    conversation_data = []
    for f in conversation_files:
        try:
            user_id = str(f.stem).split('_')[-1]
            df = pd.read_csv(f)
            if 'start_timestamp' in df.columns:
                df = df.rename(columns={'start_timestamp': 'timestamp'})
                if 'end_timestamp' in df.columns and 'timestamp' in df.columns:
                    df['duration'] = df['end_timestamp'] - df['timestamp']
                df['user_id'] = user_id
                conversation_data.append(df)
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if conversation_data:
        data['conversation'] = pd.concat(conversation_data, ignore_index=True)
    else:
        print("警告: 未找到有效的对话数据")
        data['conversation'] = pd.DataFrame(columns=['timestamp', 'duration', 'location', 'user_id'])
    
    # 加载音频数据
    print("正在加载音频数据...")
    audio_path = Path(data_dir) / 'sensing/audio'
    audio_files = list(audio_path.glob('*.csv'))
    audio_data = []
    for f in audio_files:
        try:
            user_id = str(f.stem).split('_')[-1]
            df = pd.read_csv(f)
            if 'timestamp' in df.columns:
                df['user_id'] = user_id
                if 'audio' in df.columns:
                    df = df.rename(columns={'audio': 'audio_type'})
                audio_data.append(df)
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if audio_data:
        data['audio'] = pd.concat(audio_data, ignore_index=True)
    else:
        print("警告: 未找到有效的音频数据")
        data['audio'] = pd.DataFrame(columns=['timestamp', 'audio_type', 'user_id'])
    
    # 加载蓝牙数据
    print("正在加载蓝牙数据...")
    bluetooth_path = Path(data_dir) / 'sensing/bluetooth'
    bluetooth_files = list(bluetooth_path.glob('*.csv'))
    bluetooth_data = []
    for f in bluetooth_files:
        try:
            user_id = str(f.stem).split('_')[-1]
            chunks = pd.read_csv(f, chunksize=10000)
            df = pd.concat(chunks)
            if 'time' in df.columns:
                df=df.rename(columns={'time':'timestamp'})
            else:
                print(f"警告: 文件 {f} 缺少 time 列")
                continue
            # 按timesiamp 分组统计每个时间点的设备数
            device_counts = df.groupby('timestamp').size().reset_index(name='device_count')
            device_counts['user_id'] = user_id
            bluetooth_data.append(device_counts)
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if bluetooth_data:
        data['bluetooth'] = pd.concat(bluetooth_data, ignore_index=True)
    else:
        print("警告: 未找到有效的蓝牙数据")
        data['bluetooth'] = pd.DataFrame(columns=['timestamp', 'device_count', 'user_id'])
    
    # 加载屏幕数据
    print("正在加载屏幕数据...")
    screen_path = Path(data_dir) / 'sensing/phonelock'
    screen_files = list(screen_path.glob('*.csv'))
    screen_data = []
    for f in screen_files:
        try:
            user_id = str(f.stem).split('_')[-1]
            df = pd.read_csv(f)
            if 'start' in df.columns and 'end' in df.columns:
                df['timestamp'] = df['start']
                df['duration'] = df['end'] - df['start']
                df['user_id'] = user_id
                screen_data.append(df)
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if screen_data:
        data['screen'] = pd.concat(screen_data, ignore_index=True)
    else:
        print("警告: 未找到有效的屏幕数据")
        data['screen'] = pd.DataFrame(columns=['timestamp', 'duration', 'user_id'])
    
    # 加载压力数据
    print("正在加载压力数据...")
    stress_path = Path(data_dir) / 'EMA/response/Stress'
    data['stress'] = load_stress_data(stress_path)
    
    # 基于WiFi数据中的实际位置创建POI位置映射
    if len(data['wifi']) > 0:
        unique_locations = data['wifi']['wifi_ap'].unique()
        data['poi_locations'] = pd.DataFrame({
            'wifi_ap': unique_locations,
            'poi_type': ['teaching' if any(x in loc.lower() for x in ['hall', 'library', 'lab', 'sudikoff','lsb','mclaughlin','carson-tech','sudikoff','baker-berry','presidents_house'])
                        else 'accommodation' if any(x in loc.lower() for x in ['dorm', 'residence', 'inn','hanoverrinn','dartmouth_hall','websterhall','reed'])
                        else 'eating_health' if any(x in loc.lower() for x in ['dining', 'food', 'health','venues-press','fairchild'])
                        else 'other'
                        for loc in unique_locations]
        })
    else:
        data['poi_locations'] = pd.DataFrame({
            'wifi_ap': ['library_ap', 'dorm_ap', 'dining_ap'],
            'poi_type': ['teaching', 'accommodation', 'eating_health']
        })
    
    # print(data['activity'].head(10))
    # print(data['audio'].head(10))
    # print(data['conversation'].head(10))
    # print(data['bluetooth'].head(10))
    # print(data['screen'].head(10))
    # print(data['stress'].head(10))
    # print(data['wifi'].head(10))
    # 打印数据统计信息
    print("\n数据统计:")
    for key, df in data.items():
        if isinstance(df, pd.DataFrame):
            if key == 'poi_locations':
                print(f"{key}: {len(df)} 个唯一位置")
            else:
                n_users = len(df['user_id'].unique()) if 'user_id' in df.columns else 0
                print(f"{key}: {len(df)} 条记录, {n_users} 个用户")
    
    return data

def preprocess_data(data):
    """
    预处理和对齐数据
    参数:
        data: 原始数据字典
    返回:
        X: 特征矩阵
        y: 标签
        user_ids: 用户ID列表
    """
    feature_extractor = FeatureExtractor()
    features = []
    labels = []
    user_ids = []
    
    # 检查是否有足够的数据进行处理
    if len(data['stress']) == 0:
        print("错误: 没有可用的压力数据")
        return np.array([]), np.array([]), [], []
    
    # 如果时间戳是字符串，批量转换为数值类型
    for key in data:
        if isinstance(data[key], pd.DataFrame) and 'timestamp' in data[key].columns:
            data[key]['timestamp'] = pd.to_numeric(data[key]['timestamp'], errors='coerce')
    
    # 按用户分组数据
    users = data['stress']['user_id'].unique()
    print(f"找到 {len(users)} 个有压力数据的用户")
    
    # 预处理poi_locations
    poi_locations = data['poi_locations']
    
    # 并行处理所有用户的压力测量
    from concurrent.futures import ThreadPoolExecutor
    import time
    
    start_time = time.time()
    
    def process_user(user_id):
        user_features = []
        user_labels = []
        user_ids_list = []
        
        # 获取用户的压力标签
        user_stress = data['stress'][data['stress']['user_id'] == user_id]
        
        # 获取用户的传感器数据
        user_data = {
            'wifi': data['wifi'][data['wifi']['user_id'] == user_id],
            'activity': data['activity'][data['activity']['user_id'] == user_id],
            'conversation': data['conversation'][data['conversation']['user_id'] == user_id],
            'audio': data['audio'][data['audio']['user_id'] == user_id],
            'bluetooth': data['bluetooth'][data['bluetooth']['user_id'] == user_id],
            'screen': data['screen'][data['screen']['user_id'] == user_id],
            'poi_locations': poi_locations
        }
        
        # 检查用户是否有足够的传感器数据
        if all(len(df) == 0 for df in user_data.values() if isinstance(df, pd.DataFrame) and df.columns.any() and df is not user_data['poi_locations']):
            print(f"警告: 用户 {user_id} 没有传感器数据")
            return [], [], []
        
        # 对传感器数据按时间戳排序，提高窗口查询效率
        for sensor in ['wifi', 'activity', 'conversation', 'audio', 'bluetooth', 'screen']:
            if len(user_data[sensor]) > 0 and 'timestamp' in user_data[sensor].columns:
                user_data[sensor] = user_data[sensor].sort_values('timestamp')
        
        # 对压力数据进行批量处理以减少循环次数
        valid_features = 0
        batch_size = 10  # 每批处理的压力测量数
        
        for i in range(0, len(user_stress), batch_size):
            batch = user_stress.iloc[i:i+batch_size]
            
            for _, stress_row in batch.iterrows():
                try:
                    timestamp = float(stress_row['timestamp'])
                    
                    # 获取压力测量前24小时窗口内的数据 - 使用向量化操作
                    window_start = timestamp - 24*3600
                    window_data = {
                        'wifi': user_data['wifi'][user_data['wifi']['timestamp'].between(window_start, timestamp)],
                        'activity': user_data['activity'][user_data['activity']['timestamp'].between(window_start, timestamp)],
                        'conversation': user_data['conversation'][user_data['conversation']['timestamp'].between(window_start, timestamp)],
                        'audio': user_data['audio'][user_data['audio']['timestamp'].between(window_start, timestamp)],
                        'bluetooth': user_data['bluetooth'][user_data['bluetooth']['timestamp'].between(window_start, timestamp)],
                        'screen': user_data['screen'][user_data['screen']['timestamp'].between(window_start, timestamp)],
                        'poi_locations': user_data['poi_locations']
                    }
                    
                    # 检查是否有足够的数据点 - 使用高效的长度检查
                    min_data_points = 10  # 每种传感器数据至少需要的数据点数
                    if any(len(df) >= min_data_points for df in window_data.values() if isinstance(df, pd.DataFrame) and df.columns.any() and df is not window_data['poi_locations']):
                        window_features = feature_extractor.extract_all_features(window_data)
                        user_features.append(window_features)
                        user_labels.append(stress_row['stress_level'])
                        user_ids_list.append(user_id)
                        valid_features += 1
                except Exception as e:
                    print(f"警告: 处理用户 {user_id} 的压力测量时出错: {e}")
                    continue
        
        print(f"为用户 {user_id} 提取了 {valid_features} 个有效特征集")
        return user_features, user_labels, user_ids_list
    
    # 使用线程池并行处理不同用户的数据
    all_results = []
    with ThreadPoolExecutor(max_workers=min(8, len(users))) as executor:
        all_results = list(executor.map(process_user, users))
    
    # 合并所有用户的结果
    for user_features, user_labels, user_ids_list in all_results:
        features.extend(user_features)
        labels.extend(user_labels)
        user_ids.extend(user_ids_list)
    
    end_time = time.time()
    print(f"特征提取总耗时: {end_time - start_time:.2f} 秒")
    
    if not features:
        print("错误: 无法提取有效特征")
        return np.array([]), np.array([]), [], []
    
    # 将特征转换为矩阵 - 使用numpy的向量化操作
    feature_names = list(features[0].keys())
    X = np.array([[f.get(name, 0) for name in feature_names] for f in features])
    y = np.array(labels)
    
    print(f"最终数据集形状: X: {X.shape}, y: {y.shape}")
    print(f"特征数量: {len(feature_names)}")
    print("特征名称:", feature_names)
    
    return X, y, user_ids, feature_names

def main():
    """
    运行压力预测管道的主函数
    """
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 加载数据
    print("正在加载数据...")
    data_dir = Path("dataset")
    data = load_data(data_dir)
    
    # 预处理数据
    print("正在预处理数据...")
    X, y, user_ids, feature_names = preprocess_data(data)
    
    # 将用户分为有标签和无标签集
    unique_users = np.unique(user_ids)
    n_labeled_users = len(unique_users) // 2
    labeled_users = np.random.choice(unique_users, n_labeled_users, replace=False)
    
    labeled_mask = np.isin(user_ids, labeled_users)
    X_labeled = X[labeled_mask]
    y_labeled = y[labeled_mask]
    X_unlabeled = X[~labeled_mask]
    y_unlabeled = y[~labeled_mask]  # 仅用于评估
    
    # 将有标签数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42
    )
    
    # 初始化和训练模型
    print("正在训练模型...")
    model = StressPredictor(n_estimators=100, max_depth=10)
    
    # 初始训练
    model.train_initial(X_train, y_train, X_unlabeled)
    
    # 协同训练
    print("正在进行协同训练...")
    model.co_training(X_train, y_train, X_unlabeled)
    
    # 评估
    print("正在评估模型...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n测试集性能:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 在无标签数据上评估
    unlabeled_metrics = model.evaluate(X_unlabeled, y_unlabeled)
    print("\n无标签集性能:")
    for metric, value in unlabeled_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 打印特征重要性
    if model.selected_features is not None and feature_names:
        print("\n选中的特征:")
        for i, selected in enumerate(model.selected_features):
            if selected:
                print(f"- {feature_names[i]}")

if __name__ == "__main__":
    main() 