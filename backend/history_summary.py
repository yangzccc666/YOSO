"""Small, explicitly allowlisted parameter snapshots, never credentials or logs."""

LABELS = {
    'operation': '操作', 'extraction_mode': '抽帧', 'target_fps': '张/秒',
    'interval_sec': '间隔秒', 'priority_fps': '工序张/秒', 'background_fps': '背景张/秒',
    'priority_max_images': '总上限', 'max_images_limit': '单视频上限',
    'range_start_time': '开始', 'range_end_time': '结束',
    'clip_mode': '裁剪方式', 'clip_output_name': '文件名', 'duration_start_time': '开始',
    'clip_duration': '持续时长', 'head_skip_time': '跳过开头', 'tail_skip_time': '跳过结尾',
    'remove_time_ranges': '删除时间段',
    'train_ratio': '训练比例', 'dataset_folder_name': '数据集',
    'dedup_mode': '去重模式', 'dedup_strength': '力度', 'process_strength': '工位力度',
    'limit': '数量上限', 'num_samples': '图片数量', 'model_family': '模型版本',
    'imgsz': '尺寸', 'deployment_precision': '部署精度', 'build_mode': '精度',
    'precision': '精度', 'title': '打包名', 'conf': '置信度', 'iou': 'IoU',
    'input_mode': '输入', 'annotation_mode': '标注', 'run_name': '训练名',
    'epochs': '轮次', 'batch': 'batch', 'device': '设备', 'amp': 'AMP',
}


def history_parameters(parameters):
    if not isinstance(parameters, dict):
        return {}
    values = []
    mode = parameters.get('extraction_mode')
    clip = parameters.get('clip_mode')
    for key, label in LABELS.items():
        value = parameters.get(key)
        if value is None or value == '' or isinstance(value, (dict, list)):
            continue
        if key == 'target_fps' and mode != '每秒抽取张数': continue
        if key == 'interval_sec' and mode != '按时间间隔': continue
        if key.startswith('priority_') or key == 'background_fps':
            if mode != '工序时间段高密度': continue
        if key == 'max_images_limit' and mode == '工序时间段高密度': continue
        if clip:
            groups = {'指定开始和结束时间': ('range_start_time', 'range_end_time'),
                      '指定开始时间和持续时长': ('duration_start_time', 'clip_duration'),
                      '跳过开头和结尾': ('head_skip_time', 'tail_skip_time'),
                      '删除多个时间段并拼接剩余': ('remove_time_ranges',)}
            if any(key in keys for keys in groups.values()) and key not in groups.get(clip, ()):
                continue
        values.append(f"{label}={' '.join(str(value).split())[:100]}")
    return {'关键参数': ' · '.join(values)[:1000]} if values else {}
