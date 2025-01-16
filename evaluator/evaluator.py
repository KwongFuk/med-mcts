from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
import torch


class Evaluator:
    def __init__(self, model_name="google/paligemma2-3b-pt-896"):
        """
        初始化评估器，使用 PaliGemma 模型生成评分
        :param model_name: 使用的 PaliGemma 模型名称
        """
        self.processor = PaliGemmaProcessor.from_pretrained(model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def evaluate(self, image_path, question, candidate_answer):
        """
        根据图像、问题和答案内容生成单一评分（相关性）
        :param image_path: 图像路径
        :param question: 问题文本
        :param candidate_answer: 候选答案文本
        :return: 相关性评分（0-5）
        """
        try:
            # 加载图像
            image = load_image(image_path)

            # 定义 Prompt
            prompt = (
                f"<image> Evaluate the relevance of the candidate answer based on the given question and image.\n\n"
                f"Question: {question}\n"
                f"Candidate Answer: {candidate_answer}\n\n"
                f"Provide a single relevance score in the format: Score:<score> or Relevance:<score>"
            )

            # 准备模型输入
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(torch.bfloat16).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            # 推理阶段生成评分
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # 限制生成长度
                    do_sample=False,
                )
                # 提取生成的输出
                generated_tokens = output[0][input_len:]
                decoded_output = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()
                print(f"Decoded Output: {decoded_output}")  # 输出中间结果

            # 解析评分结果
            score = self._parse_score(decoded_output)

            # 检查评分是否为有效数字
            if not isinstance(score, (int, float)) or score < 0:
                print("Invalid score detected. Setting score to 0.0.")
                score = 0.0

            return score
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 0.0  # 返回默认评分

    def _parse_score(self, output):
        """
        解析模型生成的评分结果
        :param output: 模型生成的文本
        :return: 相关性评分
        """
        try:
            # 检查可能的关键字
            if "Relevance:" in output:
                relevance = float(output.split("Relevance:")[1].strip())
                return relevance
            elif "Score:" in output:
                score = float(output.split("Score:")[1].strip())
                return score
            elif "Score :" in output:
                score = float(output.split("Score :")[1].strip())
                return score
            else:
                print("No recognizable score found in output.")
                return None
        except ValueError:
            print("Invalid score format in output.")
            return None
