from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model_path = "./Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./dataset/0.jpg",
            },
            {
                "type": "text",
                "text": "描述一下这张图片。"
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = inputs.to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128
    )

generated_ids = output_ids[:, inputs.input_ids.shape[1]:]

result = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)

print(result[0])
