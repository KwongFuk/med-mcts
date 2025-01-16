import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


class VLMModel:
    def __init__(self, model_path="deepseek-ai/deepseek-vl-7b-chat"):
        """
        初始化 DeepSeek-VL 模型和处理器
        :param model_path: 预训练 DeepSeek-VL 模型的路径
        """
        self.processor = VLChatProcessor.from_pretrained(model_path)  # 加载多模态处理器
        self.tokenizer = self.processor.tokenizer  # 获取分词器

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )  # 加载语言模型
        self.model = self.model.to(torch.bfloat16).cuda().eval()  # 将模型移动到 GPU，并设置为评估模式

    def predict(self, image_path, question=""):
        """
        使用 DeepSeek-VL 模型进行多模态推理，返回生成的响应
        :param image_path: 输入图像的路径
        :param question: 输入文本或问题
        :return: 模型生成的响应
        """
        # 准备对话结构
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{question}",  # 用户输入，包括占位符和问题
                "images": [image_path]  # 包含的图像路径
            },
            {
                "role": "Assistant",
                "content": ""  # 助手的初始响应为空
            }
        ]

        # 加载图像并准备输入
        pil_images = load_pil_images(conversation)  # 加载图像为 PIL 格式
        prepared_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)  # 将处理后的输入移动到模型设备

        # 运行图像编码器以获取图像嵌入
        inputs_embeds = self.model.prepare_inputs_embeds(**prepared_inputs)

        # 使用语言模型生成响应
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,  # 设置生成的最大 token 数
            do_sample=False,  # 禁用随机采样，使用贪婪解码
            use_cache=True  # 启用缓存以加速生成
        )

        # 解码生成的响应
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer  # 返回最终的文本响应