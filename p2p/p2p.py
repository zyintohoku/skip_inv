from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from PIL import Image
import json

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to('cuda')
        self.threshold = threshold

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to('cuda')
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to('cuda')

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to('cuda'), alphas.to('cuda')
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


torch.manual_seed(0)
NUM_DIFFUSION_STEPS = 50
LOW_RESOURCE = False
MAX_NUM_WORDS = 77
import os

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dir", type=str, required=True,
                    help="Path to folder containing inv_latents.pt (e.g., outputs/reconstruction/aidi_gs7)")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for results (default: same as latent_dir)")
parser.add_argument("--guidance_scale", type=float, default=7.0,
                    help="Guidance scale used for all timesteps when cfg_schedules.pt is absent")
parser.add_argument("--cfg_schedules_path", type=str, default=None,
                    help="Optional path to cfg_schedules.pt (overrides latent_dir/cfg_schedules.pt)")
parser.add_argument("--use_init", action="store_true",
                    help="Use init_latents.pt instead of inv_latents.pt (for upper bound)")
parser.add_argument("--start_idx", type=int, default=0,
                    help="Start index to resume from (default: 0)")
parser.add_argument("--end_idx", type=int, default=None,
                    help="End index (inclusive, default: run to end)")
args = parser.parse_args()

ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
ldm_stable.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)

tokenizer = ldm_stable.tokenizer
mapping_file_path = os.path.join(PROJECT_ROOT, "PIE_bench/mapping_file.json")
with open(mapping_file_path, "r") as f:
    editing_instruction = json.load(f)

def p2p_editing(
    latent_dir,
    output_dir=None,
    guidance_scale=7.0,
    cfg_schedules_path=None,
    use_init=False,
    start_idx=0,
    end_idx=None,
):
    if output_dir is None:
        output_dir = latent_dir
    os.makedirs(output_dir, exist_ok=True)
    
    latent_file = 'init_latents.pt' if use_init else 'inv_latents.pt'
    latents = torch.load(f'{latent_dir}/{latent_file}', weights_only=True)
    
    if cfg_schedules_path is None:
        cfg_schedules_path = f'{latent_dir}/cfg_schedules.pt'

    if os.path.exists(cfg_schedules_path):
        cfg_schedules = torch.load(cfg_schedules_path, weights_only=True)
    else:
        cfg_schedules = None
    for i,(_, item) in enumerate(editing_instruction.items()):
        if i < start_idx:
            continue
        if end_idx is not None and i > end_idx:
            break
        latent = latents[i]
        cfg_schedule = cfg_schedules[i] if cfg_schedules is not None else None
        #if cfg_schedule is not None and not all(cfg == 7 for cfg in cfg_schedule):
        #    continue
        prompt_src = item["original_prompt"].replace("[","").replace("]","")
        prompt_tgt = item["editing_prompt"].replace("[","").replace("]","")
        blended_word = item["blended_word"]
        prompts = [prompt_src, prompt_tgt]
        if blended_word != '':
            s1, s2 = blended_word.split(" ")
            blended_word = (s1, s2)
            lb = LocalBlend(prompts, blended_word)
        else:
            lb = None
        if len(prompt_src.split()) == len(prompt_tgt.split()):
            controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=0.8, self_replace_steps=0.6, local_blend=lb)
        else:
            controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=0.8, self_replace_steps=0.6, local_blend=lb)

        images, x_t = ptp_utils.text2image_ldm_stable(
            ldm_stable,
            prompts,
            controller,
            latent=latent,
            cfg_schedule=cfg_schedule,
            guidance_scale=guidance_scale,
        )
        Image.fromarray(images[0]).save(f'{output_dir}/{i}ori.png')
        Image.fromarray(images[1]).save(f'{output_dir}/{i}edi.png')

p2p_editing(
    args.latent_dir,
    args.output_dir,
    args.guidance_scale,
    args.cfg_schedules_path,
    args.use_init,
    args.start_idx,
    args.end_idx,
)
