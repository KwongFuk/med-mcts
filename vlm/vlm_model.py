# vlm/vlm_model.py
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
import torch


class VLMModel:
    def __init__(self, model_name="google/paligemma2-10b-pt-896"):
        """
        初始化 PaliGemma 模型和处理器
        :param model_name: PaliGemma 模型 ID
        """
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(model_name)

    def predict(self, image_path, question=""):
        """
        使用 PaliGemma 模型进行多模态推理，返回生成结果
        :param image_path: 输入图像路径
        :param question: 输入文本（默认为空字符串）
        :return: 模型生成的答案
        """
        # 加载图像
        image = load_image(image_path)

        # 准备模型输入
        model_inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt",
        ).to(torch.bfloat16).to(self.model.device)

        input_len = model_inputs["input_ids"].shape[-1]

        # 推理模式下生成答案
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=False,
            )
            # 生成后的新tokens
            generation = generation[0][input_len:]
            # 解码生成结果
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
