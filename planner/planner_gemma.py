from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.image_utils import load_image
from transformers import pipeline
import torch

class Planner:
    def __init__(self, model_name="google/gemma-2-2b-it"):
        """
        初始化 Planner，使用 PaliGemma 模型生成推理计划
        :param model_name: PaliGemma 模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def generate_plan(self, current_state, reflection_material, image_path):
        """
        根据反思材料和图像提供推理计划
        :param current_state: 当前状态（问题或上下文）
        :param reflection_material: 反思材料，包括祖先和兄弟节点的历史
        :param image_path: 图像路径
        :return: 推理计划（字符串列表，包含两个计划）
        """
        if not image_path:
            raise ValueError("`image_path` is required for PaliGemmaProcessor.")

        # 加载图像
        image = load_image(image_path)

        # 构造 Prompt
        prompt = (
            f"<image> Generate a reasoning plan for the following query based on the current query state "
            f"and the provided reflection materials:\n\n"
            f"Query State: {current_state}\n"
            f"Ancestor Queries: {', '.join(reflection_material['ancestor_queries'])}\n"
            f"Sibling Queries: {', '.join(reflection_material['sibling_queries'])}\n\n"
            f"Provide a reasoning plan. Only one plan."
            f"For example: 1.Identify the vehicles in the image."
        )

        # 模型输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # 使用模型生成推理计划（生成两个序列）
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,  # 限制生成长度
                do_sample=True,  # 启用采样以提高生成多样性
                temperature=0.7,  # 控制生成的多样性
                num_return_sequences=2  # 生成两个计划
            )
        print(self.tokenizer.decode(output[0]))

        # 解码生成的推理计划
        plans = [self.tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in output]

        # 提取模型生成的输出，移除 Prompt 输入部分
        extracted_plans = [self._extract_generated_plan(plan, prompt) for plan in plans]
        return extracted_plans

    def _extract_generated_plan(self, generated_output, prompt):
        """
        提取模型生成的推理计划，去除 Prompt 输入部分
        :param generated_output: 模型生成的完整输出
        :param prompt: 输入的 Prompt
        :return: 仅模型生成的部分
        """
        if prompt in generated_output:
            return generated_output.replace(prompt, "").strip()
        return generated_output
