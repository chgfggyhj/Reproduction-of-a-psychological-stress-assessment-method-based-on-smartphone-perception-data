import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

class FeatureExtractor:
    """
    特征提取器类，用于从传感器数据中提取特征
    """
    def __init__(self):
        # 定义白天和夜晚的时间范围
        self.day_start = 8  # 白天开始时间（小时）
        self.day_end = 18   # 白天结束时间（小时）
        self.window_hours = 24  # 特征提取的时间窗口（小时）

    def extract_poi_features(self, wifi_data, poi_locations):
        """
        从WiFi数据中提取位置相关特征
        参数:
            wifi_data: WiFi数据DataFrame
            poi_locations: 位置类型映射DataFrame
        返回:
            包含位置特征的字典
        """
        features = {}
        
        # 如果没有数据，返回默认值
        if len(wifi_data) == 0:
            for poi in ['teaching', 'accommodation', 'eating_health']:
                features[f'poi_{poi}_day'] = 0
                features[f'poi_{poi}_night'] = 0
            features['poi_entropy_day'] = 0
            features['poi_entropy_night'] = 0
            return features
        
        # 创建wifi_ap到poi_type的映射字典
        poi_map = dict(zip(poi_locations['wifi_ap'], poi_locations['poi_type']))
        
        # 向量化处理时间戳
        try:
            # 将时间戳转换为小时并创建新列
            wifi_data = wifi_data.copy()
            wifi_data['timestamp'] = pd.to_numeric(wifi_data['timestamp'], errors='coerce')
            wifi_data = wifi_data.dropna(subset=['timestamp'])
            
            # 转换时间戳为小时 - 使用向量化操作
            wifi_data['hour'] = pd.to_datetime(wifi_data['timestamp'], unit='s').dt.hour
            
            # 添加POI类型列
            wifi_data['poi_type'] = wifi_data['wifi_ap'].map(poi_map).fillna('unknown')
            
            # 过滤只包含关心的POI类型
            poi_types = ['teaching', 'accommodation', 'eating_health']
            valid_wifi = wifi_data[wifi_data['poi_type'].isin(poi_types)]
            
            # 使用groupby进行更快的聚合
            day_mask = (valid_wifi['hour'] >= self.day_start) & (valid_wifi['hour'] < self.day_end)
            
            # 计算白天的POI访问次数
            day_counts = valid_wifi[day_mask].groupby('poi_type').size().reindex(poi_types, fill_value=0)
            # 计算夜晚的POI访问次数
            night_counts = valid_wifi[~day_mask].groupby('poi_type').size().reindex(poi_types, fill_value=0)
            
            # 填充特征字典
            for poi in poi_types:
                features[f'poi_{poi}_day'] = day_counts.get(poi, 0)
                features[f'poi_{poi}_night'] = night_counts.get(poi, 0)

            
        except Exception as e:
            print(f"POI特征提取错误: {e}")
            # 设置默认值
            for poi in ['teaching', 'accommodation', 'eating_health']:
                features[f'poi_{poi}_day'] = 0
                features[f'poi_{poi}_night'] = 0
            features['poi_entropy_day'] = 0
            features['poi_entropy_night'] = 0
        
        return features

    def extract_activity_features(self, activity_data):
        """
        从活动数据中提取特征
        参数:
            activity_data: 活动数据DataFrame
        返回:
            包含活动特征的字典
        """
        features = {}
        
        # 如果没有数据，返回默认值
        if len(activity_data) == 0:
            for activity in ['stationary', 'walking', 'running']:
                features[f'activity_{activity}_day'] = 0
                features[f'activity_{activity}_night'] = 0
            features['activity_entropy_day'] = 0
            features['activity_entropy_night'] = 0
            return features
        
        try:
            # 向量化处理时间戳 (优化版本)
            timestamps = pd.to_numeric(activity_data['timestamp'], errors='coerce')
            valid_mask = ~np.isnan(timestamps)
            hours = np.full(len(activity_data), -1, dtype=np.int32)
            hours[valid_mask] = pd.to_datetime(timestamps[valid_mask], unit='s').dt.hour.values
            activity_data = activity_data.assign(hour=hours)
            
            # 过滤有效活动类型
            activity_types = ['stationary', 'walking', 'running']
            valid_activity = activity_data[activity_data['activity'].isin(activity_types)]
            
            if len(valid_activity) == 0:
                # 没有有效活动数据
                for activity in activity_types:
                    features[f'activity_{activity}_day'] = 0
                    features[f'activity_{activity}_night'] = 0
                features['activity_entropy_day'] = 0
                features['activity_entropy_night'] = 0
                return features
            
            # 使用向量化操作划分白天和夜晚
            day_mask = (valid_activity['hour'] >= self.day_start) & (valid_activity['hour'] < self.day_end)
            
            # 计算白天活动次数
            day_counts = valid_activity[day_mask].groupby('activity').size().reindex(activity_types, fill_value=0)
            # 计算夜晚活动次数
            night_counts = valid_activity[~day_mask].groupby('activity').size().reindex(activity_types, fill_value=0)
            
            # 填充特征
            for activity in activity_types:
                features[f'activity_{activity}_day'] = day_counts.get(activity, 0)
                features[f'activity_{activity}_night'] = night_counts.get(activity, 0)
            
        except Exception as e:
            print(f"活动特征提取错误: {e}")
            # 设置默认值
            for activity in ['stationary', 'walking', 'running']:
                features[f'activity_{activity}_day'] = 0
                features[f'activity_{activity}_night'] = 0
            features['activity_entropy_day'] = 0
            features['activity_entropy_night'] = 0
        
        return features

    def extract_conversation_features(self, conversation_data, audio_data):
        """
        从对话和音频数据中提取特征
        参数:
            conversation_data: 对话数据DataFrame
            audio_data: 音频数据DataFrame
        返回:
            包含对话和音频特征的字典
        """
        features = {}
        
        # 处理音频数据 - 向量化操作
        if len(audio_data) > 0:
            try:
                # 优化音频数据处理
                timestamps = pd.to_numeric(audio_data['timestamp'], errors='coerce')
                valid_mask = ~np.isnan(timestamps)
                hours = np.full(len(audio_data), -1, dtype=np.int32)
                hours[valid_mask] = pd.to_datetime(timestamps[valid_mask], unit='s').dt.hour.values
                audio_data = audio_data.assign(hour=hours).drop(columns=['timestamp'])
                
                # 确保audio_type列存在并处理无效值
                if 'audio_type' not in audio_data.columns:
                    audio_data['audio_type'] = 'unknown'
                
                # 定义音频类型和检测白天/夜晚
                audio_types = ['silence', 'voice', 'noise']
                day_mask = (audio_data['hour'] >= self.day_start) & (audio_data['hour'] < self.day_end)
        
                # 过滤有效音频类型
                valid_audio = audio_data[audio_data['audio_type'].isin(audio_types)]
                
                # 使用groupby聚合
                day_audio_counts = valid_audio.loc[day_mask].groupby('audio_type').size().reindex(audio_types, fill_value=0)
                night_audio_counts = valid_audio.loc[~day_mask].groupby('audio_type').size().reindex(audio_types, fill_value=0)
                
                # 填充特征
                for audio_type in audio_types:
                    features[f'audio_{audio_type}_day'] = day_audio_counts.get(audio_type, 0)
                    features[f'audio_{audio_type}_night'] = night_audio_counts.get(audio_type, 0)
                
            except Exception as e:
                print(f"音频特征提取错误: {e}")
                for audio_type in ['silence', 'voice', 'noise']:
                    features[f'audio_{audio_type}_day'] = 0
                    features[f'audio_{audio_type}_night'] = 0
        else:
            # 没有音频数据时的默认值
            for audio_type in ['silence', 'voice', 'noise']:
                features[f'audio_{audio_type}_day'] = 0
                features[f'audio_{audio_type}_night'] = 0
        
        # 处理对话数据 - 向量化操作
        if len(conversation_data) > 0:
            try:
                # 清理和准备数据
                conversation_data = conversation_data.copy()
                conversation_data['timestamp'] = pd.to_numeric(conversation_data['timestamp'], errors='coerce')
                conversation_data = conversation_data.dropna(subset=['timestamp'])
                
                # 添加小时列进行时间分段
                conversation_data['hour'] = pd.to_datetime(conversation_data['timestamp'], unit='s').dt.hour
                
                # 确保duration列存在
                if 'duration' not in conversation_data.columns:
                    conversation_data['duration'] = 0
                else:
                    conversation_data['duration'] = pd.to_numeric(conversation_data['duration'], errors='coerce').fillna(0)
                
                # # 确保location列存在
                # if 'location' not in conversation_data.columns:
                #     conversation_data['location'] = 'unknown'
                
                # 检测白天/夜晚
                day_mask = (conversation_data['hour'] >= self.day_start) & (conversation_data['hour'] < self.day_end)
        
                # 计算白天/夜晚对话统计量
                features['conv_count_day'] = day_mask.sum()
                features['conv_count_night'] = (~day_mask).sum()
                features['conv_duration_day'] = conversation_data.loc[day_mask, 'duration'].sum()
                features['conv_duration_night'] = conversation_data.loc[~day_mask, 'duration'].sum()

            except Exception as e:
                print(f"对话特征提取错误: {e}")
                # 设置默认值
                features['conv_count_day'] = 0
                features['conv_duration_day'] = 0
                features['conv_count_night'] = 0
                features['conv_duration_night'] = 0
                for poi in ['teaching', 'accommodation', 'eating_health']:
                    features[f'conv_count_{poi}'] = 0
                    features[f'conv_duration_{poi}'] = 0
        else:
            # 没有对话数据时的默认值
            features['conv_count_day'] = 0
            features['conv_duration_day'] = 0
            features['conv_count_night'] = 0
            features['conv_duration_night'] = 0
            for poi in ['teaching', 'accommodation', 'eating_health']:
                features[f'conv_count_{poi}'] = 0
                features[f'conv_duration_{poi}'] = 0
            
        return features

    def extract_bluetooth_features(self, bluetooth_data):
        """
        提取蓝牙传感器特征
        参数:
            bluetooth_data: 蓝牙数据DataFrame
        返回:
            包含蓝牙特征的字典
        """
        # 定义POI类型列表
        poi_types = ['teaching', 'accommodation', 'eating_health']
        features = {}
        
        if bluetooth_data is None or bluetooth_data.empty:
            # 默认特征
            features['bluetooth_devices_day'] = 0
            features['bluetooth_devices_night'] = 0
            for poi in poi_types:
                features[f'bluetooth_devices_{poi}'] = 0
            return features

        try:
            # 预处理：转换时间戳并提取小时
            bluetooth_data = bluetooth_data.copy()
            bluetooth_data['timestamp'] = pd.to_numeric(bluetooth_data['timestamp'], errors='coerce')
            bluetooth_data = bluetooth_data.dropna(subset=['timestamp'])
            bluetooth_data['hour'] = pd.to_datetime(bluetooth_data['timestamp'], unit='s').dt.hour

            # 按白天和夜晚统计扫描到的设备数量
            day_mask = (bluetooth_data['hour'] >= self.day_start) & (bluetooth_data['hour'] < self.day_end)
            features['bluetooth_devices_day'] = bluetooth_data[day_mask].shape[0]
            features['bluetooth_devices_night'] = bluetooth_data[~day_mask].shape[0]


        except Exception as e:
            print(f"蓝牙特征提取错误: {e}")
            features['bluetooth_devices_day'] = 0
            features['bluetooth_devices_night'] = 0
            for poi in poi_types:
                features[f'bluetooth_devices_{poi}'] = 0

        return features

    def extract_screen_features(self, screen_data):
        """
        提取屏幕传感器特征（睡眠相关）
        参数:
            screen_data: 屏幕数据DataFrame
        返回:
            包含睡眠特征的字典
        """
        features = {}
        if screen_data is None or screen_data.empty:
            features['sleep_duration'] = 0
            features['sleep_start_hour'] = 0
            return features

        try:
            # 检查是否有锁屏事件列
            if 'event' not in screen_data.columns:
                # 如果没有event列，尝试使用所有数据
                screen_data = screen_data.copy()
                screen_data['duration'] = pd.to_numeric(screen_data['duration'], errors='coerce')
                screen_data = screen_data[screen_data['duration'] >= 3600]  # 持续1小时以上
            else:
                # 只考虑锁屏事件，且持续时间超过1小时（认为是睡眠）
                screen_data = screen_data.copy()
                screen_data = screen_data[screen_data['event'] == 'lock']
                screen_data['duration'] = pd.to_numeric(screen_data['duration'], errors='coerce')
                screen_data = screen_data[screen_data['duration'] >= 3600]  # 持续1小时以上

            if screen_data.empty:
                features['sleep_duration'] = 0
                features['sleep_start_hour'] = 0
                return features

            # 找到持续时间最长的事件作为睡眠
            longest_sleep = screen_data.loc[screen_data['duration'].idxmax()]
            features['sleep_duration'] = longest_sleep['duration'] / 3600.0  # 转换为小时
            # 计算入睡时间（开始时间戳）
            start_timestamp = longest_sleep['timestamp'] - longest_sleep['duration']
            start_hour = pd.to_datetime(start_timestamp, unit='s').hour
            features['sleep_start_hour'] = start_hour

        except Exception as e:
            print(f"睡眠特征提取错误: {e}")
            features['sleep_duration'] = 0
            features['sleep_start_hour'] = 0

        return features

    def extract_relative_features(self, absolute_features, user_history):
        """
        提取相对特征（相对于用户历史数据的特征）
        参数:
            absolute_features: 绝对特征字典
            user_history: 用户历史特征字典
        返回:
            包含相对特征的字典
        """
        features = absolute_features.copy()
        
        if user_history is None or not user_history:
            return features
        
        # 预先计算所有历史均值和标准差以避免重复计算
        history_stats = {}
        for feature in user_history:
            values = np.array(user_history[feature])
            if len(values) > 0:
                history_stats[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        # 批量计算Z分数
        for feature, value in absolute_features.items():
            if feature in history_stats:
                mean = history_stats[feature]['mean']
                std = history_stats[feature]['std']
                
                if std > 0:
                    features[f'{feature}_zscore'] = (value - mean) / std
                else:
                    features[f'{feature}_zscore'] = 0
                        
        return features

    def extract_all_features(self, data_window, user_history=None):
        """
        提取所有特征（优化版）
        参数:
            data_window: 包含所有传感器数据的字典
            user_history: 用户历史特征字典（可选）
        返回:
            包含所有特征的字典
        """
        features = {}
        

        
        processed_data = {}
        with ThreadPoolExecutor(max_workers=6) as executor:  # 增加工作线程数
            # 预处理所有传感器
            sensors = [s for s in ['wifi', 'activity', 'conversation', 'audio', 'bluetooth', 'screen'] 
                      if data_window.get(s) is not None]
                      
            # 批量提交预处理任务
            futures = {sensor: executor.submit(self._preprocess_sensor_data, data_window[sensor]) 
                      for sensor in sensors}
            
            # 快速获取POI数据（不经过预处理）
            poi_locations = data_window.get('poi_locations')
            
            # 收集预处理结果
            for sensor, future in futures.items():
                processed_data[sensor] = future.result()

        # 2. 并行特征提取阶段
        with ThreadPoolExecutor(max_workers=6) as executor:  # 增加工作线程数
            # 提交所有特征提取任务
            extract_tasks = {
                'poi': executor.submit(self._extract_poi_parallel, processed_data.get('wifi'), poi_locations),
                'activity': executor.submit(self.extract_activity_features, processed_data.get('activity')),
                'conversation': executor.submit(self._extract_conversation_parallel,
                                              processed_data.get('conversation'),
                                              processed_data.get('audio')),
                'bluetooth': executor.submit(self.extract_bluetooth_features, processed_data.get('bluetooth')),
                'screen': executor.submit(self.extract_screen_features, processed_data.get('screen'))
            }

            # 批量获取结果
            for name, task in extract_tasks.items():
                try:
                    features.update(task.result() or {})
                except Exception as e:
                    print(f"特征提取错误[{name}]: {str(e)[:50]}...")  # 简化错误输出
                    features.update(self._get_default_features(name))

        # 3. 后处理阶段
        if user_history is not None:
            with ThreadPoolExecutor() as executor:
                rel_features = executor.submit(self.extract_relative_features, features, user_history)
                features = rel_features.result()
            
        return features

    def _preprocess_sensor_data(self, sensor_data):
        """预处理传感器数据（通用方法）"""
        if sensor_data is None or sensor_data.empty:
            return pd.DataFrame()
            
        try:
            # 统一处理时间戳列
            df = sensor_data.copy()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
            return df
        except Exception as e:
            print(f"传感器数据预处理错误: {str(e)[:50]}...")
            return pd.DataFrame()

    def _extract_poi_parallel(self, wifi_data, poi_locations):
        """并行处理POI特征提取"""
        return self.extract_poi_features(
            wifi_data if wifi_data is not None else pd.DataFrame(),
            poi_locations if poi_locations is not None else pd.DataFrame()
        )

    def _extract_conversation_parallel(self, conversation_data, audio_data):
        """并行处理对话特征提取"""
        return self.extract_conversation_features(
            conversation_data if conversation_data is not None else pd.DataFrame(),
            audio_data if audio_data is not None else pd.DataFrame()
        )

    def _get_default_features(self, feature_type):
        """快速生成各类型默认特征"""
        defaults = {
            'poi': (
                {f'poi_{poi}_day':0 for poi in ['teaching','accommodation','eating_health']} 
                | {f'poi_{poi}_night':0 for poi in ['teaching','accommodation','eating_health']}
            ),
            'activity': (
                {f'activity_{act}_day':0 for act in ['stationary','walking','running']}
                | {f'activity_{act}_night':0 for act in ['stationary','walking','running']}
            ),
            'conversation': (
                {f'audio_{t}_day':0 for t in ['silence','voice','noise']}
                | {f'audio_{t}_night':0 for t in ['silence','voice','noise']}
                | {'conv_count_day':0, 'conv_duration_day':0, 'conv_count_night':0, 'conv_duration_night':0}
                | {f'conv_count_{poi}':0 for poi in ['teaching','accommodation','eating_health']}
                | {f'conv_duration_{poi}':0 for poi in ['teaching','accommodation','eating_health']}
            ),
            'bluetooth': {
                'bluetooth_devices_day': 0,
                'bluetooth_devices_night': 0,
                **{f'bluetooth_devices_{poi}':0 for poi in ['teaching','accommodation','eating_health']}
            },
            'screen': {
                'sleep_duration': 0,
                'sleep_start_hour': 0
            }
        }
        return defaults.get(feature_type, {})

    def _create_default_poi_features(self):
        """快速生成默认POI特征"""
        features = {}
        # 白天POI特征
        features.update({f'poi_{poi}_day':0 for poi in ['teaching','accommodation','eating_health']})
        # 夜晚POI特征
        features.update({f'poi_{poi}_night':0 for poi in ['teaching','accommodation','eating_health']})
        # 熵特征
        features.update({'poi_entropy_day':0, 'poi_entropy_night':0})
        return features
