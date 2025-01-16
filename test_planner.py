from planner.planner import Planner

def main():
    # 初始化 Planner
    planner = Planner(model_name="google/paligemma2-3b-pt-896")

    # 示例输入
    current_state = "What is the color of the car?"
    reflection_material = {
        "ancestor_queries": ["Describe the objects in the image.", "Identify vehicles in the image."],
        "sibling_queries": ["What is the size of the car?", "Where is the car located?"]
    }
    image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"  # 替换为实际图像路径

    # 生成推理计划
    plans = planner.generate_plan(current_state, reflection_material, image_path)
    print("Generated Plans:")
    for i, plan in enumerate(plans, start=1):
        print(f"Plan {i}:\n{plan}\n")

if __name__ == "__main__":
    main()

