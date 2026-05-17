from pathlib import Path

# 直接使用你代码中的路径
model_path = '/absolute/path/to/face_landmarker.task'

# 创建一个 Path 对象并检查
model_file = Path(model_path)

if model_file.is_file():
    print(f"✓ 模型文件存在，大小: {model_file.stat().st_size} 字节")
else:
    print(f"✗ 错误：在路径 '{model_path}' 找不到模型文件。请确认文件已下载并放置正确。")