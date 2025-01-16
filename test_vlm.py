from vlm.vlm_model import VLMModel

# 初始化模型
vlm = VLMModel()

# 图像路径和问题
image_path = "/home/MCTS-PPO/image/Lok1_SG_LMP_image.png"  
question = "<image> Review the Histopathology image and pick the opyion the base aligns with the severity illustrated. Options: A.malignant B.benign Please select the corrext answer from the options above."

# 模型推理
answer = vlm.predict(image_path, question)
print(f"Model Answer: {answer}")
