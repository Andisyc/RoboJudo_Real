import onnxruntime as ort

# 替换成你的模型路径
model_path = "./assets/models/g1/beyondmimic/Violin.onnx" # g1_dance

try:
    sess = ort.InferenceSession(model_path)
    meta = sess.get_modelmeta()
    custom_map = meta.custom_metadata_map
    
    print("===== ONNX Metadata =====")
    for key, value in custom_map.items():
        # 有些训练代码会把长度存为 'num_frames' 或类似的名字
        print(f"{key}: {value}")
        print("\n")
        
    # 如果 metadata 里没有直接写帧数，你可以看看有没有提示
    if "default_joint_pos" in custom_map:
        print("\n[Check] Model metadata loaded successfully.")
    else:
        print("\n[Warning] No BeyondMimic metadata found.")

except Exception as e:
    print(f"Error loading model: {e}")