import pandas as pd
import numpy as np
from pathlib import Path
from feature_extraction import FeatureExtractor
from model import StressPredictor
from sklearn.model_selection import train_test_split
import json



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
                if ' end_timestamp' in df.columns and 'timestamp' in df.columns:
                    df['duration'] = df[' end_timestamp'] - df['timestamp']
                df['user_id'] = user_id
                conversation_data.append(df)
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if conversation_data:
        data['conversation'] = pd.concat(conversation_data, ignore_index=True)
    else:
        print("警告: 未找到有效的对话数据")
        data['conversation'] = pd.DataFrame(columns=['timestamp', 'duration', 'user_id'])
    
    # 加载音频数据
    print("正在加载音频数据...")
    audio_path = Path(data_dir) / 'sensing/audio'
    audio_files = list(audio_path.glob('*.csv'))
    audio_data = []
    for f in audio_files:
        try:
            user_id = str(f.stem).split('_')[-1]
            df = pd.read_csv(f)
            if ' audio inference' in df.columns:
                df = df.rename(columns={' audio inference': 'audio_type'})
            audio_map={0:'silence',1:'voice',2:'noise',3:'unknown'}
            df['audio_type'] = df['audio_type'].map(audio_map).fillna('unknown')
            if 'timestamp' in df.columns:
                df['user_id']=user_id
            audio_data.append(df)
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if audio_data:
        data['audio'] = pd.concat(audio_data, ignore_index=True)
    else:
        print("警告: 未找到有效的音频数据")
        data['audio'] = pd.DataFrame(columns=['timestamp', 'audio_type', 'user_id'])
    
    # 加载蓝牙数据（保留原始设备记录）
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
                df = df.rename(columns={'time': 'timestamp'})
            elif 'timestamp' not in df.columns:
                print(f"警告: 文件 {f} 缺少时间戳列")
                continue
            
            # 保留原始蓝牙扫描记录
            df['user_id'] = user_id
            bluetooth_data.append(df[['timestamp', 'user_id']])  # 只需时间戳和用户ID
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if bluetooth_data:
        data['bluetooth'] = pd.concat(bluetooth_data, ignore_index=True)
    else:
        print("警告: 未找到有效的蓝牙数据")
        data['bluetooth'] = pd.DataFrame(columns=['timestamp', 'user_id'])
    
    # 加载屏幕数据（添加事件类型）
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
                df['event'] = 'lock'  # 添加事件类型列
                df['user_id'] = user_id
                screen_data.append(df)
        except Exception as e:
            print(f"警告: 加载文件 {f} 时出错: {e}")
    if screen_data:
        data['screen'] = pd.concat(screen_data, ignore_index=True)
    else:
        print("警告: 未找到有效的屏幕数据")
        data['screen'] = pd.DataFrame(columns=['timestamp', 'duration', 'event', 'user_id'])
    
    # 加载压力数据（修复格式处理）
    print("正在加载压力数据...")
    stress_path = Path(data_dir) / 'EMA/response/Stress'
    stress_data = []
    stress_files = list(stress_path.glob('*.json'))
    
    for f in stress_files:
        try:
            # 从文件名中提取用户ID
            user_id = str(f.stem).split('_')[1]
            with open(f, 'r') as file:
                json_content = json.load(file)  # 重命名变量避免冲突
                # 处理不同格式的压力数据
                if isinstance(json_content, list):
                    for entry in json_content:
                        if isinstance(entry, dict):
                            # 提取时间戳和压力水平
                            timestamp = entry.get('resp_time', entry.get('timestamp', 0))
                            level = entry.get('level')
                            
                            if timestamp and level is not None:
                                try:
                                    stress_level = int(level)
                                    if stress_level in {1,2,3}:
                                        stress_level=1
                                    elif stress_level in {4,5}:
                                        stress_level=0
                                    else:
                                        stress_level='unknown'
                                    stress_data.append({
                                        'timestamp': timestamp,
                                        'stress_level': stress_level,
                                        'user_id': user_id
                                    })
                                except (ValueError, TypeError):
                                    continue
                elif isinstance(json_content, dict):
                    # 处理单条记录格式
                    timestamp = json_content.get('resp_time', json_content.get('timestamp', 0))
                    level = json_content.get('level')
                    if timestamp and level is not None:
                        try:
                            stress_level = int(level)
                            if stress_level in {1,2,3}:
                                stress_level=1
                            elif stress_level in {4,5}:
                                stress_level=0
                            else:
                                stress_level='unknown'
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
        data['stress'] = pd.DataFrame(stress_data)
    else:
        print("警告: 未找到有效的压力数据")
        data['stress'] = pd.DataFrame(columns=['timestamp', 'stress_level', 'user_id'])
    
    # 基于WiFi数据中的实际位置创建POI位置映射
    if len(data['wifi']) > 0:
        unique_locations = data['wifi']['wifi_ap'].unique()
        data['poi_locations'] = pd.DataFrame({
            'wifi_ap': unique_locations,
            'poi_type': ['teaching' if any(x in loc.lower() for x in ['hall', 'library', 'lab', 'sudikoff','lsb','mclaughlin','carson-tech','sudikoff','baker-berry','presidents_house','baker-berry','steele','burke','rollins-chapel','ropeferry','vail','remsen','parkhurst','hanoverpsych','kemeny','moore','batrlett','thornton','remote_offices_HREAP','maclean','kellogg','sanborn','cummings','murdough','tllc','ripley','presidents_house','french'])
                        else 'accommodation' if any(x in loc.lower() for x in ['dorm', 'residence', 'inn','hanoverrinn','dartmouth_hall','websterhall','reed','east-wheelock','topliff','fayerweather','Mckenzie','north-park','lodge','massrow','butterfield','fahey-mclane','gile','newhamp','maxwell','streeter','buchanan','lord','hitchcock','Cohen','channing-cox','richardson','bissell','evergreen'])
                        else 'eating_health' if any(x in loc.lower() for x in ['dining', 'food', 'health','venues-press','fairchild','college-street','hopkins','robinson','53_commons','7-lebanon','sport-venues','softballfield','berry_sports_center','occum','external','aquinas'])
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
    预处理和对齐数据（优化内存使用）
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
    unlabel_features=[]
    
    # 检查是否有足够的数据进行处理
    if len(data['stress']) == 0:
        print("错误: 没有可用的压力数据")
        return np.array([]), np.array([]), [], []
    
    # 按用户分组数据
    users = data['stress']['user_id'].unique()
    print(f"找到 {len(users)} 个有压力数据的用户")
    
    # 预处理poi_locations
    poi_locations = data['poi_locations']
    
    # 优化内存使用：按用户加载数据
    for user_id in users:
        print(f"处理用户: {user_id}")
        
        # 获取用户的压力标签
        user_stress = data['stress'][data['stress']['user_id'] == user_id].copy()
        
        # 按需加载用户数据，减少内存占用
        user_data = {}
        for sensor in ['wifi', 'activity', 'conversation', 'audio', 'bluetooth', 'screen']:
            if sensor in data and 'user_id' in data[sensor].columns:
                # 使用query优化内存使用
                user_data[sensor] = data[sensor].query(f"user_id == '{user_id}'").copy()
            else:
                user_data[sensor] = pd.DataFrame()
        
        # 添加POI位置映射
        user_data['poi_locations'] = poi_locations
        
        # 如果时间戳是字符串，转换为数值类型
        for sensor in ['wifi', 'activity', 'conversation', 'audio', 'bluetooth', 'screen']:
            if not user_data[sensor].empty and 'timestamp' in user_data[sensor].columns:
                user_data[sensor]['timestamp'] = pd.to_numeric(
                    user_data[sensor]['timestamp'], errors='coerce')
                # 按时间戳排序
                user_data[sensor] = user_data[sensor].sort_values('timestamp')
        
        # 检查用户是否有足够的传感器数据
        if all(len(df) == 0 for df in user_data.values() if isinstance(df, pd.DataFrame) and df.columns.any() and df is not user_data['poi_locations']):
            print(f"警告: 用户 {user_id} 没有传感器数据")
            continue
        
        user_features = []
        user_labels = []
        user_unlabel_features=[]
        

        # 处理每个压力测量点
        for _, stress_row in user_stress.iterrows():
            try:
                timestamp = float(stress_row['timestamp'])
                # 获取压力测量前24小时窗口内的数据
                window_start = timestamp - 24*3600
                window_end = timestamp
                
                # 创建时间窗口数据字典
                window_data = {}
                for sensor in ['wifi', 'activity', 'conversation', 'audio', 'bluetooth', 'screen']:
                    if sensor in user_data and not user_data[sensor].empty:
                        sensor_df = user_data[sensor]
                        # 过滤时间窗口内的数据
                        mask = (sensor_df['timestamp'] >= window_start) & (sensor_df['timestamp'] <= window_end)
                        window_data[sensor] = sensor_df[mask].copy()
                    else:
                        window_data[sensor] = pd.DataFrame()
                
                # 添加POI位置映射
                window_data['poi_locations'] = user_data['poi_locations']
                # 检查是否有足够的数据点
                min_data_points = 10  # 每种传感器数据至少需要的数据点数
                if any(len(df) >= min_data_points for df in window_data.values() if isinstance(df, pd.DataFrame) and df.columns.any() and df is not window_data['poi_locations']):
                    window_features = feature_extractor.extract_all_features(window_data)
                    user_features.append(window_features)
                    user_labels.append(stress_row['stress_level'])
                
            except Exception as e:
                print(f"警告: 处理用户 {user_id} 的压力测量时出错: {e}")
                continue

        # 添加到总数据集
        features.extend(user_features)
        labels.extend(user_labels)
        user_ids.extend([user_id] * len(user_features))
        
        print(f"为用户 {user_id} 提取了 {len(user_features)} 个有效特征集")
        if len(user_stress) > 0:
            unlabel_number=max(0,100-len(user_stress))
            if unlabel_number>0:
                print(f"为用户{user_id}生成{unlabel_number}组未标记数据")
                first_stress_time=user_stress['timestamp'].min()
                for i in range(1,unlabel_number+1):
                    Window_start=first_stress_time+i*24*3600
                    Window_end=Window_start+24*3600
                    # 创建时间窗口数据字典
                    Window_data = {}
                    for sensor in ['wifi', 'activity', 'conversation', 'audio', 'bluetooth', 'screen']:
                        if sensor in user_data and not user_data[sensor].empty:
                            sensor_df = user_data[sensor]
                            # 过滤时间窗口内的数据
                            mask = (sensor_df['timestamp'] >= Window_start) & (sensor_df['timestamp'] <= Window_end)
                            Window_data[sensor] = sensor_df[mask].copy()
                        else:
                            Window_data[sensor] = pd.DataFrame()
                
                    # 添加POI位置映射
                    Window_data['poi_locations'] = user_data['poi_locations']
                    # 检查是否有足够的数据点
                    min_data_points = 10  # 每种传感器数据至少需要的数据点数
                    if any(len(df) >= min_data_points for df in Window_data.values() if isinstance(df, pd.DataFrame) and df.columns.any() and df is not Window_data['poi_locations']):
                        Window_features = feature_extractor.extract_all_features(Window_data)
                        user_unlabel_features.append(Window_features)

        unlabel_features.extend(user_unlabel_features)
                
    if not features:
        print("错误: 无法提取有效特征")
        return np.array([]), np.array([]), [], []
    
    # 将特征转换为矩阵
    feature_names = list(features[0].keys()) if features else []
    X_unlabel=np.array([[f.get(name, 0) for name in feature_names] for f in unlabel_features])
    X = np.array([[f.get(name, 0) for name in feature_names] for f in features])
    y = np.array(labels)
    
    print(f"最终数据集形状: X: {X.shape}, y: {y.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    return X, y, user_ids, feature_names,X_unlabel

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
    X, y, user_ids, feature_names,X_unlabeled = preprocess_data(data)
    
    
    
    # 将有标签数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X , y, test_size=0.2, random_state=42
    )
    
    # 初始化和训练模型
    print("正在训练模型...")
    model = StressPredictor(n_estimators=100, max_depth=10)
    
    # 初始训练（使用无标签数据的特征，不使用其标签）
    model.train_initial(X_train, y_train, X_unlabeled)
    
    # 协同训练（使用无标签数据的特征，不使用其标签）
    print("正在进行协同训练...")
    model.co_training(X_train, y_train, X_unlabeled)
    
    # 评估测试集性能
    print("正在评估测试集性能...")
    test_metrics = model.evaluate(X_test,y_test)
    
    print("\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 保存训练好的模型
    import joblib
    model_path = "trained_model.pkl"
    joblib.dump(model, model_path)
    print(f"模型已保存到 {model_path}")
    
    # 打印特征重要性
    if model.selected_features is not None and feature_names:
        print("\n选中的特征:")
        for i, selected in enumerate(model.selected_features):
            if selected:
                print(f"- {feature_names[i]}")

if __name__ == "__main__":
    main()
