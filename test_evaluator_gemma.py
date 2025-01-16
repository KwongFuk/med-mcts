from evaluator.evaluator_gemma import Evaluator

def main():
    evaluator = Evaluator(model_name="google/gemma-2-2b-it")
    
    # 示例输入
    image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"  # 替换为本地实际图像路径
    question = "What is the color of the car?"
    candidate_answer = "The car is blue."
    candidate_answer1 = "green."

    # 评估单一评分
    relevance_score = evaluator.evaluate(image_path, question, candidate_answer1)

    if relevance_score is not None:
        print(f"Relevance Score: {relevance_score}")
    else:
        print("Failed to evaluate relevance score.")

if __name__ == "__main__":
    main()

