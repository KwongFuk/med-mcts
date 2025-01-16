from planner.planner_gemma import Planner

def main():
    # 初始化 Planner
    planner = Planner(model_name="google/gemma-2-2b-it")

    # 示例输入
    current_state = "was patient 12724975 diagnosed with hypoxemia until 1 year ago and did a chest x-ray reveal any tubes/lines in the abdomen during the same period?"
    reflection_material = {
        "ancestor_queries": ["Locate the objects of interest.", "Identify the regions of interest in the image."],
        "sibling_queries": ["Analyze their relationships and attributes.", "Identify the objects that are relevant to the query."]
    }
    # image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"  # 替换为实际图像路径

    # 生成推理计划
    plans = planner.generate_plan(current_state, reflection_material, image_path)
    print("Generated Plans:")
    for i, plan in enumerate(plans, start=1):
        print(f"*Plan {i}:\n{plan}\n")

if __name__ == "__main__":
    main()

