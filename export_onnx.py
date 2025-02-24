#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import os
import sys
import torch
import argparse
from tqdm import tqdm
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from einops import rearrange, repeat

torch.set_grad_enabled(False)
parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('-m', '--model_path', type=str, default="deepseek-ai/deepseek-vl2-tiny", help='path to the torch model.')
parser.add_argument('-s', '--seq_length', type=int, default=3072, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cuda")
parser.add_argument('-f', '--folder', type=str, default='./models/onnx')
args = parser.parse_args()

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    dtype = torch.bfloat16

processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
tokenizer = processor.tokenizer

origin_model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
origin_model = origin_model.to(dtype).to(device).eval()

folder = args.folder
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/vit')

for param in origin_model.parameters():
    param.requires_grad = False

llm = origin_model.language
mlp = origin_model.projector
vit = origin_model.vision
config = llm.config
transformer = llm.model
layers = transformer.layers
image_newline = origin_model.image_newline
view_seperator = origin_model.view_seperator
global_view_pos = origin_model.config.global_view_pos

# text config
IMAGE_SIZE = 384
SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size
MASK_VALUE = -3.3895e+38 # -10000.0
print(f'Layers: {NUM_LAYERS}\n\
Hidden size: {HIDDEN_SIZE}\n\
Query heads: {NUM_ATTENTION_HEADS}\n\
KV heads: {NUM_KEY_VALUE_HEADS}\n')

class VisionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pixel_values):
        images = vit(pixel_values)
        images_embeds = mlp(images)
        return images_embeds


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.embed_tokens(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos.to(device)
        self.sin.to(device)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            position_embeddings=(self.cos, self.sin),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos.to(device)
        self.sin.to(device)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            position_embeddings=(self.cos, self.sin),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states) 
        m_logits = llm.lm_head(hidden_states)
        return m_logits


class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token

# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class PenaltySampleHead(torch.nn.Module):

    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, m_logits, input_ids, top_p, temperature, penalty):
        # repeat penalty
        logits = torch.gather(m_logits, 1, input_ids)
        logits = torch.where(logits < 0, logits * penalty, logits / penalty)
        m_logits.scatter_(1, input_ids, logits)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


def convert_vision_transformer():
    model = VisionTransformer()
    pixel_values = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE)).to(dtype)

    torch.onnx.export(
        model, (pixel_values),
        f'{folder}/vit/vision_transformer.onnx',
        verbose=False,
        input_names=["pixel_values"],
        output_names=['images_embed'],
        do_constant_folding=True,
    )

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.ones(1, 1, SEQ_LENGTH, SEQ_LENGTH).to(dtype).to(device)
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)

    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')

def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, HIDDEN_SIZE).to(dtype).to(device)

    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')

def convert_greedy_head():   
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE).to(dtype).to(device)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)

def convert_penalty_sample_head():   
    model = PenaltySampleHead()
    m_logits = torch.randn(1, VOCAB_SIZE).to(dtype).to(device)
    input_ids = torch.tensor([range(SEQ_LENGTH)]).to(device)
    top_p = torch.tensor([0.8]).to(device)
    temperature = torch.tensor([0.98]).to(device)
    penalty = torch.tensor([0.98]).to(device)

    torch.onnx.export(
        model, (m_logits, input_ids, top_p, temperature, penalty),
        f'{folder}/penalty_sample_head.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'top_p', 'temperature',
            'penalty'
        ],
        output_names=['probs', 'token'],
        do_constant_folding=True,
        opset_version=15)

def test_net_with_mask(image_path):
    query = "描述一下这张图片,图片中存在人头,如果存在,看一下他嘴里是否叼着香烟或者笔、棒棒糖、筷子等其他物品。图片中是否有手,如果有,看一下手里拿的是不是香烟或者笔、粉笔、棒棒糖、筷子等其他物品。并结合周围的场景,判断出图片中是否有人在抽烟。"
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{query}",
            "images": [image_path],
        },
        {
            "role": "<|Assistant|>",
            "content": ""
        },
    ]
    print(f'image: "{image_path}"')
    print(f'query: "{query}"\n')

    pil_images = load_pil_images(conversation)
    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(device)

    # init inputs
    token_len = inputs['input_ids'].shape[1]
    input_ids = torch.zeros(SEQ_LENGTH).unsqueeze(0).to(torch.int32).to(device)
    input_ids[:,:token_len] = inputs['input_ids'].to(torch.int32)
    input_ids[input_ids < 0] = 0 # ignore the image embeddings
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    # float implement
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float().to(device) * MASK_VALUE # -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH)

    # init models
    vision = VisionTransformer()
    embed = Embedding()
    lm_head = LmHead()
    greedy_head = GreedyHead()
    blocks = []
    block_kvs = []
    for i in range(NUM_LAYERS):
        blocks.append(Block(i))
        block_kvs.append(BlockCache(i))

    # inference
    if inputs.images is None or inputs.images_spatial_crop.sum() == 0:
        out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    else:
        bs, max_n_images, _ = inputs.images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = inputs.images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += (1 + num_width_tiles * num_height_tiles)

            total_tiles.append(inputs.images[idx, :batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        assert total_tiles.shape[0] != 0

        out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
        # multi-batch multi-time influence precision
        """
        images_embeds = []
        for batch_i in range(total_tiles.shape[0]):
            images_embed = vision(total_tiles[batch_i:batch_i+1].to(dtype))
            images_embeds.append(images_embed)
        images_embed = torch.cat(images_embeds, dim=0)
        """
        # multi-batch multi-time original implement
        images_embed = vision(total_tiles.to(dtype))

        # vision postprocess
        _, hw, n_dim = images_embed.shape
        h = w = int(hw ** 0.5)
        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        for idx in range(inputs.images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(inputs.images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = inputs.images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embed[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embed[tile_index + 1: tile_index + 1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1

                # format global and local features

                # ----------------- global view add newline -----------------
                # [hw, D] -> [h, w, D]
                global_features = global_features.view(h, w, n_dim)
                # [D]     -> [h, 1, D]
                new_lines_in_global = repeat(image_newline, "d -> h 1 d", h=h)
                # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                # [h, w + 1, D] -> [h * (w + 1), D]
                global_features = global_features.view(-1, n_dim)

                # ----------------- local view add newline -----------------
                # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                local_features = rearrange(
                    local_features,
                    "(th tw) (h w) d -> (th h) (tw w) d",
                    th=num_height_tiles,
                    tw=num_width_tiles,
                    h=h,
                    w=w
                )

                # [D] -> [num_height_tiles * h, 1, D]
                new_lines_in_local = repeat(
                    image_newline,
                    "d -> (th h) 1 d",
                    th=num_height_tiles,
                    h=h
                )

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                local_features = local_features.view(-1, n_dim)

                # ----------------- merge global and local tiles -----------------
                if global_view_pos == "head":
                    global_local_features = torch.cat(
                        [global_features, view_seperator[None, :], local_features], dim=0)
                else:
                    global_local_features = torch.cat(
                        [local_features, view_seperator[None, :], global_features], dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                out[idx].masked_scatter_(torch.cat((inputs.images_seq_mask[idx].unsqueeze(-1), torch.tensor([[False]]*(SEQ_LENGTH-token_len), device=device)), dim=0), images_in_this_batch)

    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out, position_ids, attention_mask)
        k_cache.append(k)
        v_cache.append(v)
    out = out[:, token_len - 1: token_len].view(1, 1, HIDDEN_SIZE)
    token = greedy_head(lm_head(out)).view(1)
    out_ids = []
    while int(token) != processor.tokenizer.eos_token_id:
        out_ids.append(int(token))
        word = processor.tokenizer.decode([int(token)])
        print(word, end="")
        token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = MASK_VALUE # -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out, position_ids, attention_mask, k_cache[i], v_cache[i])
            k_cache[i][:,token_len-1:token_len] = k
            v_cache[i][:,token_len-1:token_len] = v
        token = greedy_head(lm_head(out)).view(1)
    print("\noutput_ids:{}".format(out_ids))

test_net_with_mask('datasets/YAN/BGSSSS0001.jpg')

"""
print(f'Convert vision transformer')
convert_vision_transformer()

print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_cache(i)
 
print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()
print("Done")
"""