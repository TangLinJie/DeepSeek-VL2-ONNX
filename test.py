import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
# vl_gpt = vl_gpt.to(torch.bfloat16).cpu().eval()

## single image conversation example
## Please note that <|ref|> and <|/ref|> are designed specifically for the object localization feature. These special tokens are not required for normal conversations.
## If you would like to experience the grounded captioning functionality (responses that include both object localization and reasoning), you need to add the special token <|grounding|> at the beginning of the prompt. Examples could be found in Figure 9 of our paper.
"""
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n<|ref|>The giraffe at the back.<|/ref|>.",
        "images": ["./images/visual_grounding_1.jpeg"],
    },
    {"role": "<|Assistant|>", "content": ""},
]
"""
import os
root = "/home/ljtang/DeepSeek-VL2/datasets/YAN"
w = os.walk(root)
acc = 0
total = 0
for (dirpath, dirnames, filenames) in w:
    filenames = sorted(filenames)
    for filename in filenames:
        if str(filename).endswith(".json"):
          continue
        print("file: ", os.path.join(str(dirpath), str(filename)))
        conversation = [
            {
                "role": "<|User|>",
                # "content": "<image>\n<|ref|>香烟.<|/ref|>.",
                # "content": "描述一下这张图片,图片中存在人头,如果存在,看一下他嘴里是否叼着香烟或者笔、棒棒糖、筷子等其他物品。图片中是否有手,如果有,看一下手里拿的是不是香烟或者笔、粉笔、棒棒糖、筷子等其他物品。并结合周围的场景,判断出图片中是否有人在抽烟。",
                # "content": "<|grounding|>第一张图片包含了香烟，请识别是否有相同的类别对象在第二张图片。",
                # "images": [os.path.join(str(dirpath), "cy0001.jpg"), os.path.join(str(dirpath), str(filename))],
                "content": "<image>\n描述一下这张图片,图片中存在人头,如果存在,看一下他嘴里是否叼着香烟或者笔、棒棒糖、筷子等其他物品。图片中是否有手,如果有,看一下手里拿的是不是香烟或者笔、粉笔、棒棒糖、筷子等其他物品。并结合周围的场景,判断出图片中是否有人在抽烟。",
                "images": [os.path.join(str(dirpath), str(filename))],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        if str(dirpath).split('/')[-1] == '1' and "<|det|>" in answer:
            acc += 1
        if str(dirpath).split('/')[-1] == '0' and "<|det|>" not in answer:
            acc += 1
        total += 1
        print("\n")
print("acc: ", acc/total)