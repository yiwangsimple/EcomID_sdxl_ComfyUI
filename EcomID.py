import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import math
import cv2
import PIL.Image
from .resampler import Resampler
from .CrossAttentionPatch import Attn2Replace, instantid_attention, pulid_attention
from .utils import tensor_to_image

from insightface.app import FaceAnalysis
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper


try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F
from torch import nn

MODELS_DIR = os.path.join(folder_paths.models_dir, "instantid")
if "instantid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["instantid"]
folder_paths.folder_names_and_paths["instantid"] = (current_paths, folder_paths.supported_pt_extensions)

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

from .eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from .encoders import IDEncoder

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

MODELS_DIR = os.path.join(folder_paths.models_dir, "pulid")
if "pulid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["pulid"]
folder_paths.folder_names_and_paths["pulid"] = (current_paths, folder_paths.supported_pt_extensions)

class PulidModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.image_proj_model = self.init_id_adapter()
        self.image_proj_model.load_state_dict(model["image_proj"])
        self.ip_layers = To_KV(model["ip_adapter"])

    def init_id_adapter(self):
        image_proj_model = IDEncoder()
        return image_proj_model

    def get_image_embeds(self, face_embed, clip_embeds):
        embeds = self.image_proj_model(face_embed, clip_embeds)
        return embeds

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor

def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1] * (source.dim() - 1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

def to_gray(img):
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.repeat(1, 3, 1, 1)
    return x

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class InstantID(torch.nn.Module):
    def __init__(self, instantid_model, cross_attention_dim=1280, output_cross_attention_dim=1024, clip_embeddings_dim=512, clip_extra_context_tokens=16):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(instantid_model["image_proj"])
        self.ip_layers = To_KV(instantid_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        #image_prompt_embeds = clip_embed.clone().detach()
        image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = clip_embed_zeroed.clone().detach()
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)

        return image_prompt_embeds, uncond_image_prompt_embeds

class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = torch.nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value

def _set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()
    
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(pulid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(pulid_attention, **patch_kwargs)

class InstantID_IPA_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "instantid_file": (folder_paths.get_filename_list("instantid"), )}}

    RETURN_TYPES = ("INSTANTID",)
    FUNCTION = "load_model"
    CATEGORY = "EcomID"

    def load_model(self, instantid_file):
        ckpt_path = folder_paths.get_full_path("instantid", instantid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        model = InstantID(
            model,
            cross_attention_dim=1280,
            output_cross_attention_dim=model["ip_adapter"]["1.to_k_ip.weight"].shape[1],
            clip_embeddings_dim=512,
            clip_extra_context_tokens=16,
        )

        return (model,)

def extractFeatures(insightface, image, extract_kps=False):
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640) # reset the detection size

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(face_img[i])
            if face:
                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

                if extract_kps:
                    out.append(draw_kps(face_img[i], face['kps']))
                else:
                    out.append(torch.from_numpy(face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        else:
            out = torch.stack(out, dim=0)
    else:
        out = None

    return out

######
'''
node
'''
class EcomID_PulidModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pulid_file": (folder_paths.get_filename_list("pulid"), )}}

    RETURN_TYPES = ("PULID",)
    FUNCTION = "load_model"
    CATEGORY = "EcomID"

    def load_model(self, pulid_file):
        ckpt_path = folder_paths.get_full_path("pulid", pulid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        return (model,)

class EcomIDEvaClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("EVA_CLIP",)
    FUNCTION = "load_eva_clip"
    CATEGORY = "EcomID"

    def load_eva_clip(self):
        from .eva_clip.factory import create_model_and_transforms

        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)

        model = model.visual

        eva_transform_mean = getattr(model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            model["image_mean"] = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            model["image_std"] = (eva_transform_std,) * 3

        return (model,)

class EcomIDFaceAnalysis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insight_face"
    CATEGORY = "EcomID"

    def load_insight_face(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)

class FaceKeypointsPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "faceanalysis": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image"
    CATEGORY = "EcomID"

    def preprocess_image(self, faceanalysis, image):
        face_kps = extractFeatures(faceanalysis, image, extract_kps=True)

        if face_kps is None:
            face_kps = torch.zeros_like(image)
            print(f"\033[33mWARNING: no face detected, unable to extract the keypoints!\033[0m")
            #raise Exception('Face Keypoints Image: No face detected.')

        return (face_kps,)

def add_noise(image, factor):
    seed = int(torch.sum(image).item()) % 1000000007
    torch.manual_seed(seed)
    mask = (torch.rand_like(image) < factor).float()
    noise = torch.rand_like(image)
    noise = torch.zeros_like(image) * (1-mask) + noise * mask

    return factor*noise

class ApplyEcomID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid_ipa": ("INSTANTID", ),
                "pulid": ("PULID", ),
                "eva_clip": ("EVA_CLIP",),
                "insightface": ("FACEANALYSIS", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "method": (["fidelity", "style", "neutral"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "image_kps": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("MODEL", "positive", "negative", )
    FUNCTION = "apply_EcomID"
    CATEGORY = "EcomID"

    def apply_EcomID(self, instantid_ipa, pulid, eva_clip, insightface, control_net, image, model, positive, negative, start_at, end_at, weight=.8, ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average',
                        method=None, fidelity=None, projection=None):
        self.dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        self.device = comfy.model_management.get_torch_device()

        ip_weight = weight if ip_weight is None else ip_weight
        cn_strength = weight if cn_strength is None else cn_strength

        face_embed = extractFeatures(insightface, image)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        # if no keypoints image is provided, use the image itself (only the first one in the batch)
        face_kps = extractFeatures(insightface, image_kps if image_kps is not None else image[0].unsqueeze(0), extract_kps=True)

        if face_kps is None:
            face_kps = torch.zeros_like(image) if image_kps is None else image_kps
            print(f"\033[33mWARNING: No face detected in the keypoints image!\033[0m")

        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            #clip_embed_zeroed = add_noise(clip_embed, noise)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        # 1: patch the attention
        self.instantid = instantid_ipa
        self.instantid.to(self.device, dtype=self.dtype)

        # 提取第一种embedding
        image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        work_model = model.clone()

        if mask is not None:
            mask = mask.to(self.device)

        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        eva_clip.to(device, dtype=dtype)
        pulid_model = PulidModel(pulid).to(device, dtype=dtype)

        if mask is not None:
            if mask.dim() > 3:
                mask = mask.squeeze(-1)
            elif mask.dim() < 3:
                mask = mask.unsqueeze(0)
            mask = mask.to(device, dtype=dtype)

        if method == "fidelity" or projection == "ortho_v2":
            num_zero = 8
            ortho = False
            ortho_v2 = True
        elif method == "style" or projection == "ortho":
            num_zero = 16
            ortho = True
            ortho_v2 = False
        else:
            num_zero = 0
            ortho = False
            ortho_v2 = False

        if fidelity is not None:
            num_zero = fidelity

        # face_analysis.det_model.input_size = (640,640)
        image = tensor_to_image(image)

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=device,
        )

        face_helper.face_parse = None
        face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        cond = []
        uncond = []

        for i in range(image.shape[0]):
            # get insightface embeddings
            iface_embeds = None
            for size in [(size, size) for size in range(640, 256, -64)]:
                insightface.det_model.input_size = size
                face = insightface.get(image[i])
                if face:
                    face = sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[
                        -1]
                    iface_embeds = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
                    break
            else:
                raise Exception('insightface: No face detected.')

            # get eva_clip embeddings
            face_helper.clean_all()
            face_helper.read_image(image[i])
            face_helper.get_face_landmarks_5(only_center_face=True)
            face_helper.align_warp_face()

            if len(face_helper.cropped_faces) == 0:
                raise Exception('facexlib: No face detected.')

            face = face_helper.cropped_faces[0]
            face = image_to_tensor(face).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            parsing_out = \
            face_helper.face_parse(T.functional.normalize(face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(face)
            face_features_image = torch.where(bg, white_image, to_gray(face))
            face_features_image = T.functional.resize(face_features_image, eva_clip.image_size,
                                                      T.InterpolationMode.BICUBIC).to(device, dtype=dtype)
            face_features_image = T.functional.normalize(face_features_image, eva_clip.image_mean, eva_clip.image_std)

            id_cond_vit, id_vit_hidden = eva_clip(face_features_image, return_all_features=False, return_hidden=True,
                                                  shuffle=False)
            id_cond_vit = id_cond_vit.to(device, dtype=dtype)
            for idx in range(len(id_vit_hidden)):
                id_vit_hidden[idx] = id_vit_hidden[idx].to(device, dtype=dtype)

            id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

            # combine embeddings
            id_cond = torch.cat([iface_embeds, id_cond_vit], dim=-1)
            if noise == 0:
                id_uncond = torch.zeros_like(id_cond)
            else:
                id_uncond = torch.rand_like(id_cond) * noise
            id_vit_hidden_uncond = []
            for idx in range(len(id_vit_hidden)):
                if noise == 0:
                    id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[idx]))
                else:
                    id_vit_hidden_uncond.append(torch.rand_like(id_vit_hidden[idx]) * noise)
            # 提取第二种embedding
            cond.append(pulid_model.get_image_embeds(id_cond, id_vit_hidden))
            uncond.append(pulid_model.get_image_embeds(id_uncond, id_vit_hidden_uncond))

        # average embeddings
        cond = torch.cat(cond).to(device, dtype=dtype)
        uncond = torch.cat(uncond).to(device, dtype=dtype)
        if cond.shape[0] > 1:
            cond = torch.mean(cond, dim=0, keepdim=True)
            uncond = torch.mean(uncond, dim=0, keepdim=True)

        if num_zero > 0:
            if noise == 0:
                zero_tensor = torch.zeros((cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device)
            else:
                zero_tensor = torch.rand((cond.size(0), num_zero, cond.size(-1)), dtype=dtype, device=device) * noise
            cond = torch.cat([cond, zero_tensor], dim=1)
            uncond = torch.cat([uncond, zero_tensor], dim=1)

        sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(end_at)

        patch_kwargs = {
            "pulid": pulid_model,
            "weight": ip_weight,
            "cond": cond,
            "uncond": uncond,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "ortho": ortho,
            "ortho_v2": ortho_v2,
            "mask": mask,
        }

        number = 0
        for id in [4, 5, 7, 8]:  # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number * 2 + 1)
                _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6):  # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number * 2 + 1)
                _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number * 2 + 1)
            _set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
            number += 1

        # 2: do the ControlNet
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device()) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device())

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False

        return(work_model, cond_uncond[0], cond_uncond[1], )

class ApplyEcomIDAdvanced(ApplyEcomID):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid_ipa": ("INSTANTID", ),
                "pulid": ("PULID",),
                "eva_clip": ("EVA_CLIP",),
                "insightface": ("FACEANALYSIS", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "method": (["fidelity", "style", "neutral"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "ip_weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 3.0, "step": 0.01, }),
                "cn_strength": ("FLOAT", {"default": .8, "min": 0.0, "max": 10.0, "step": 0.01, }),
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, }),
                "combine_embeds": (['average', 'norm average', 'concat'], {"default": 'average'}),
            },
            "optional": {
                "image_kps": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

class InstantIDAttentionPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid": ("INSTANTID", ),
                "insightface": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "model": ("MODEL", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 3.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "FACE_EMBEDS")
    FUNCTION = "patch_attention"
    CATEGORY = "EcomID"

    def patch_attention(self, instantid, insightface, image, model, weight, start_at, end_at, noise=0.0, mask=None):
        self.dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        self.device = comfy.model_management.get_torch_device()

        face_embed = extractFeatures(insightface, image)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)
        if clip_embed.shape[0] > 1:
            clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        # 1: patch the attention
        self.instantid = instantid
        self.instantid.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        if weight == 0:
            return (model, { "cond": image_prompt_embeds, "uncond": uncond_image_prompt_embeds } )

        work_model = model.clone()

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        if mask is not None:
            mask = mask.to(self.device)

        patch_kwargs = {
            "weight": weight,
            "ipadapter": self.instantid,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "mask": mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        number = 0
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
            number += 1

        return(work_model, { "cond": image_prompt_embeds, "uncond": uncond_image_prompt_embeds }, )

class ApplyInstantIDControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_embeds": ("FACE_EMBEDS", ),
                "control_net": ("CONTROL_NET", ),
                "image_kps": ("IMAGE", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative", )
    FUNCTION = "apply_controlnet"
    CATEGORY = "EcomID"

    def apply_controlnet(self, face_embeds, control_net, image_kps, positive, negative, strength, start_at, end_at, mask=None):
        self.device = comfy.model_management.get_torch_device()

        if strength == 0:
            return (positive, negative)

        if mask is not None:
            mask = mask.to(self.device)

        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        image_prompt_embeds = face_embeds['cond']
        uncond_image_prompt_embeds = face_embeds['uncond']

        cnets = {}
        cond_uncond = []
        control_hint = image_kps.movedim(-1,1)

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_at, end_at))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device()) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device())

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False

        return(cond_uncond[0], cond_uncond[1])

NODE_CLASS_MAPPINGS = {
    "InstantID_IPA_ModelLoader": InstantID_IPA_ModelLoader,
    "EcomID_PulidModelLoader": EcomID_PulidModelLoader,
    "EcomIDEvaClipLoader": EcomIDEvaClipLoader,
    "EcomIDFaceAnalysis": EcomIDFaceAnalysis,
    "ApplyEcomID": ApplyEcomID,
    "ApplyEcomIDAdvanced": ApplyEcomIDAdvanced,
    "FaceKeypointsPreprocessor": FaceKeypointsPreprocessor,

    "InstantIDAttentionPatch": InstantIDAttentionPatch,
    "ApplyInstantIDControlNet": ApplyInstantIDControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantID_IPA_ModelLoader": "Load InstantID Ipa Model (EcomID)",
    "EcomIDFaceAnalysis": "EcomID Face Analysis",
    "EcomID_PulidModelLoader": "Load PuLID Model (EcomID)",
    "EcomIDEvaClipLoader": "Load Eva Clip (EcomID)",
    "ApplyEcomID": "Apply EcomID",
    "ApplyEcomIDAdvanced": "Apply EcomID Advanced",
    "FaceKeypointsPreprocessor": "Face Keypoints Preprocessor",

    "InstantIDAttentionPatch": "InstantID Patch Attention",
    "ApplyInstantIDControlNet": "InstantID Apply ControlNet",
}
