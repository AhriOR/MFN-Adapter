import json
import time
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPImageProcessor, AutoModel, AutoImageProcessor
from PIL import Image
from typing import Tuple, Dict, Union, Optional

from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup, AutoProcessor
from peft import get_peft_model, LoraConfig
import os


def parse_args():
    parser = argparse.ArgumentParser(description='PFU-Adapter')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0')

    # Path
    parser.add_argument('--new_model_save_path', type=str, default='./checkpoints/best_model.pth',
                        help='Model_Save_Path')
    parser.add_argument('--bert_tokenizer_path', type=str, default='./tokenizer/bert_tokenizer', help='BERT_Model_Path')
    parser.add_argument('--clip_image_processor_path', type=str, default='./tokenizer/clip_image_processor', help='CLIP_Image_Processor_Path')
    parser.add_argument('--bert_model_path', type=str, default='./bert_model', help='BERT_Model_Path')
    parser.add_argument('--vit_model_path', type=str, default='./vit_model', help='ViT_Model_Path')
    parser.add_argument('--qwen_model_path', type=str, default='./GLM/glm-edge-v-2b', help='Qwen_Model_Path')

    # Training Parameter
    parser.add_argument('--batch_size', type=int, default=16, help='Batch_Size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Num_Epochs')
    parser.add_argument('--info_nce_temperature', type=float, default=0.07, help='InfoNCE_Temperature')
    parser.add_argument('--contrat_loss_weight', type=float, default=0.2, help='Contrastive_Loss_Weight')
    parser.add_argument('--lr', type=float, default=8e-5, help='Learning_Rate')
    parser.add_argument('--weight_decay', type=float, default=2e-5, help='Weight_Decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.01, help='Warmup_Ratio')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max_Grad_Norm')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Accumulation_Steps')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Warmup_Ratio')

    # Model Parameter
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding_Dim')
    parser.add_argument('--qwen_hidden_size', type=int, default=2048, help='Qwen_Hidden_Size')
    parser.add_argument('--fusion_input_dim', type=int, default=512, help='Fusion_Input_Dim')

    # LORA Parameter
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA_R')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA_Alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA_Dropout')

    # Focal Loss Parameter
    parser.add_argument('--focal_alpha', type=float, default=0.5, help='Focal_Alpha')
    parser.add_argument('--focal_gamma', type=float, default=1, help='Focal_Gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label_Smoothing')

    args = parser.parse_args()
    return args



args = parse_args()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


EMBED_DIM = args.embed_dim
QWEN_HIDDEN_SIZE = args.qwen_hidden_size
FUSION_INPUT_DIM = args.fusion_input_dim
MODEL_SAVE_PATH = args.model_save_path
NEW_MODEL_SAVE_PATH = args.new_model_save_path
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = args.num_epochs
INFO_NCE_TEMPERATURE = nn.Parameter(torch.tensor(args.info_nce_temperature, device=device))
CONTRAT_LOSS_WEIGHT = args.contrat_loss_weight
LOSS_WEIGHT = torch.tensor([1.0, 1.0]).to(device)


try:
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_path, trust_remote_code=True)
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.clip_image_processor_path,
                                                              trust_remote_code=True)
    BERT_MODEL_PATH = args.bert_model_path
    VIT_MODEL_PATH = args.vit_model_path
    from dataloader import train_dataloader, test_dataloader
except Exception as e:
    print(f"Warning: Failed to load tokenizers/processors or dataloaders: {e}. Using placeholder paths.")


# Encoders & Fusion
class EncoderClassifier(nn.Module):
    """ Projection Head """

    def __init__(self, input_dim, embed_dim):
        super(EncoderClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.classifier(x)



class QwenEncoder(nn.Module):
    """Qwen3-0.6B Encoder"""

    def __init__(self, embed_dim):
        super(QwenEncoder, self).__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained('./Qwen/Qwen3-VL-2B-Instruct', trust_remote_code=True)

        for param in self.qwen.parameters():
            param.requires_grad = False

        qwen_output_dim = self.qwen.config.text_config.hidden_size
        self.projection = EncoderClassifier(qwen_output_dim, embed_dim, ratio=2)
        self.patch_projection = nn.Linear(qwen_output_dim, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (B, L, 2048)
        raw_patch_features = hidden[:, 1:, :]
        cls_features = self.projection(hidden[:, 0, :])

        projected_patch_features = self.patch_projection(raw_patch_features)

        return cls_features, projected_patch_features


class VitImageEncoder(nn.Module):

    def __init__(self, embed_dim, vit_pretrained="./vit_model"):
        super(VitImageEncoder, self).__init__()

        self.vit_processor = CLIPImageProcessor.from_pretrained('./tokenizer/clip_image_processor',
                                                                trust_remote_code=True)

        self.vit_processor.size = {"height": 224, "width": 224}
        self.vit_processor.do_resize = True
        self.vit_processor.do_center_crop = True


        self.vit_model = AutoModel.from_pretrained(
            vit_pretrained,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )


        self.vit_hidden_size = self.vit_model.config.hidden_size

        self.projection_head = EncoderClassifier(self.vit_hidden_size, embed_dim, ratio=2)
        self.patch_projection = nn.Linear(self.vit_hidden_size, embed_dim)

    def forward(self, image):
        """
        Input：PIL.Image List
        Output：projected_patch_features - [B, L, embed_dim]
        """

        valid_images = []
        placeholder_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        for img in image:
            if img is None or not isinstance(img, Image.Image):
                valid_images.append(placeholder_img)
            else:
                valid_images.append(img)

        vit_inputs = self.vit_processor(
            images=valid_images,
            return_tensors="pt",
        )

        vit_inputs = {k: v.to(next(self.vit_model.parameters()).device) for k, v in vit_inputs.items()}


        with torch.no_grad():
            vit_outputs = self.vit_model(**vit_inputs)

        raw_patch_features = vit_outputs.last_hidden_state

        cls_features = self.projection_head(raw_patch_features[:, 0, :])
        return cls_features, self.patch_projection(raw_patch_features[:, 1:, :])


class ClipInfoNCEloss(nn.Module):

    def __init__(self, temperature=INFO_NCE_TEMPERATURE):
        super(ClipInfoNCEloss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, image_feat, text_feat, match_label):
        """
        image_feat: (B, D)
        text_feat: (B, D)
        match_label: (B,)
        """
        B = image_feat.shape[0]

        image_feat = F.normalize(image_feat, dim=-1, eps=self.eps)
        text_feat = F.normalize(text_feat, dim=-1, eps=self.eps)

        # ---------------Similairty Matrix---------------
        # Img2Txt
        sim_img2txt = torch.matmul(image_feat, text_feat.t()) / self.temperature
        # Txt2Img
        sim_txt2img = sim_img2txt.t()

        sim_img2txt = torch.clamp(sim_img2txt, min=-50.0, max=50.0)
        sim_txt2img = torch.clamp(sim_txt2img, min=-50.0, max=50.0)

        # --------------- Match Matrix ---------------
        label_matrix = torch.eye(B, dtype=torch.long).to(sim_img2txt.device)
        label_matrix[torch.arange(B), torch.arange(B)] = (1 - match_label).long()

        label_matrix_t = label_matrix.t()

        # --------------- Mask ---------------
        pos_mask_img2txt = label_matrix == 1
        neg_mask_img2txt = label_matrix == 0

        pos_mask_txt2img = label_matrix_t == 1
        neg_mask_txt2img = label_matrix_t == 0


        exp_sim_img2txt = torch.exp(sim_img2txt)
        pos_sum_img2txt = (exp_sim_img2txt * pos_mask_img2txt).sum(dim=1)
        neg_sum_img2txt = (exp_sim_img2txt * neg_mask_img2txt).sum(dim=1)

        numerator_img2txt = pos_sum_img2txt + self.eps
        denominator_img2txt = pos_sum_img2txt + neg_sum_img2txt + self.eps
        loss_img2txt = -torch.log(numerator_img2txt / denominator_img2txt)
        loss_img2txt = torch.clamp(loss_img2txt, max=50.0)


        exp_sim_txt2img = torch.exp(sim_txt2img)
        pos_sum_txt2img = (exp_sim_txt2img * pos_mask_txt2img).sum(dim=1)
        neg_sum_txt2img = (exp_sim_txt2img * neg_mask_txt2img).sum(dim=1)

        numerator_txt2img = pos_sum_txt2img + self.eps
        denominator_txt2img = pos_sum_txt2img + neg_sum_txt2img + self.eps
        loss_txt2img = -torch.log(numerator_txt2img / denominator_txt2img)
        loss_txt2img = torch.clamp(loss_txt2img, max=50.0)


        valid_mask_img2txt = pos_sum_img2txt > 0

        valid_mask_txt2img = pos_sum_txt2img > 0

        valid_mask = valid_mask_img2txt & valid_mask_txt2img

        if valid_mask.sum() == 0:

            return torch.tensor(0.0, device=sim_img2txt.device, requires_grad=True)


        loss_img2txt_valid = loss_img2txt[valid_mask].mean()
        loss_txt2img_valid = loss_txt2img[valid_mask].mean()
        total_loss = (loss_img2txt_valid + loss_txt2img_valid) / 2

        return total_loss


class QwenGatedSelfAttention(nn.Module):
    """Qwen Gated Self Attention"""

    def __init__(self, embedding_dim, num_heads=8, gate_dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads


        assert self.head_dim * num_heads == embedding_dim

        self.attn_proj = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )


        self.gate_proj_head = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)
        ])


        self.gate_dropout = nn.Dropout(gate_dropout)


        for gate in self.gate_proj_head:
            nn.init.ones_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, q, k, v, mask=None, h_pre=None):

        B, L, D = q.shape

        attn_output, _ = self.attn_proj(q, k, v, attn_mask=mask)

        if h_pre is None:
            h_pre = q

        h_pre_split = h_pre.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output_split = attn_output.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        gated_output_split = []
        for i in range(self.num_heads):
            head_h_pre = h_pre_split[:, i, :, :]
            head_attn = attn_output_split[:, i, :, :]

            head_gate_logits = self.gate_proj_head[i](head_h_pre)
            head_gate_scores = torch.sigmoid(head_gate_logits)
            head_gate_scores = self.gate_dropout(head_gate_scores)

            head_gated = head_attn * head_gate_scores
            gated_output_split.append(head_gated)

        gated_output = torch.stack(gated_output_split, dim=1)
        gated_output = gated_output.transpose(1, 2).contiguous().view(B, L, D)

        return gated_output


class GatedFusion(nn.Module):
    def __init__(self, input_dim, num_feats=5):
        super(GatedFusion, self).__init__()
        self.input_dim = input_dim
        self.num_feats = num_feats

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.RMSNorm(input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_feats)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat_list):
        """
        Args:
            feat_list:  [img_private, txt_private, img_pooled, txt_pooled, fusion_feat]

        Returns:
            gated_feats
        """
        gated_feats = []

        for idx, feat in enumerate(feat_list):
            if feat.dim() == 3:
                feat_flat = feat.squeeze(1)
            elif feat.dim() == 2:
                feat_flat = feat
            else:
                raise ValueError(f"特征维度需为 2 或 3，当前为 {feat.dim()}")


            weight = self.gates[idx](feat_flat)

            if idx == 0:
                all_weights = [weight]
            else:
                all_weights.append(weight)

        # 权重归一化（可选）
        all_weights = torch.cat(all_weights, dim=1)
        norm_weights = self.softmax(all_weights)

        # 重新遍历，应用归一化后的权重
        for idx, feat in enumerate(feat_list):
            weight = norm_weights[:, idx:idx + 1]

            if feat.dim() == 3:
                weight = weight.unsqueeze(1)
            # 加权特征（广播机制）
            gated_feat = weight * feat
            gated_feats.append(gated_feat)

        return gated_feats


class PrivateFeatureClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class Extraction(nn.Module):
    """Orthogonal feature extractor"""

    def __init__(self, embedding_dim=QWEN_HIDDEN_SIZE, num_classes=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.proj_head = PrivateFeatureClassifier(embedding_dim, num_classes)
        self.ortho_gamma = 2.0

    def forward(self, img_pooled, txt_pooled, label, eps=1e-6):

        img = img_pooled.squeeze(1)  # (B, D)
        txt = txt_pooled.squeeze(1)  # (B, D)


        txt_dir = F.normalize(txt, dim=-1, eps=eps)  # (B, D)
        img_dir = F.normalize(img, dim=-1, eps=eps)  # (B, D)

        #  Calculate the "cross-modal projection coefficient
        #  (the component of the current modality in the direction of the other modality)

        proj_i2t = torch.sum(img * txt_dir, dim=-1, keepdim=True) + eps
        proj_t2i = torch.sum(txt * img_dir, dim=-1, keepdim=True) + eps


        img_private = img - proj_i2t * txt_dir
        txt_private = txt - proj_t2i * img_dir

        img_private_norm = F.normalize(img_private, dim=-1, eps=eps)
        txt_private_norm = F.normalize(txt_private, dim=-1, eps=eps)
        # 逐样本内积 → 绝对值 → 批次均值（标量损失）
        inner_product = ((img_private_norm * txt_private_norm).sum(dim=-1)) ** 2

        img_prediction = self.proj_head(img_private)
        txt_prediction = self.proj_head(txt_private)

        # KL_Div
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        with torch.no_grad():
            img_conf = F.softmax(img_prediction, dim=-1).max(dim=-1).values
            txt_conf = F.softmax(txt_prediction, dim=-1).max(dim=-1).values
            img_pred_label = img_prediction.argmax(dim=-1)
            txt_pred_label = txt_prediction.argmax(dim=-1)
            disagree_mask = (img_pred_label != txt_pred_label)
            agree_mask = (
                (img_pred_label == txt_pred_label)
            )

        ortho_weight = torch.ones_like(inner_product)
        ortho_weight[disagree_mask] = self.ortho_gamma

        ortho_loss = (ortho_weight * inner_product).mean()

        if agree_mask.sum() > 0:
            kl_div_loss = kl_loss_fn(
                F.log_softmax(img_prediction[agree_mask], dim=-1),
                F.softmax(txt_prediction[agree_mask], dim=-1)
            )
        else:
            kl_div_loss = torch.tensor(0.0, device=img_prediction.device)

        ce_loss_fn = nn.CrossEntropyLoss()
        cross_entropy_loss = (ce_loss_fn(img_prediction, label) + ce_loss_fn(txt_prediction, label)) / 2

        img_private = img_private.unsqueeze(1)
        txt_private = txt_private.unsqueeze(1)

        return img_private, txt_private, ortho_loss, kl_div_loss, cross_entropy_loss


class Fusion(nn.Module):
    """Fusion Module"""

    def __init__(self, embedding_dim, num_classes, ffn_hidden_ratio=4, dropout=0.4, loss_weight=None):
        super(Fusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes


        self.self_attn = QwenGatedSelfAttention(embedding_dim, num_heads=8)
        self.self_attn_before_fusion = nn.MultiheadAttention(embedding_dim, num_heads=8,
                                                             batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * ffn_hidden_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * ffn_hidden_ratio, embedding_dim)
        )

        self.gated_fusion = GatedFusion(embedding_dim)

        self.img2txt_weight = nn.Parameter(torch.tensor(0.0))

        self.pre_gated_norm = nn.LayerNorm(embedding_dim)
        self.fusion_res_norm = nn.LayerNorm(embedding_dim)
        self.cross_pool = nn.Linear(embedding_dim, 1)
        self.extraction = Extraction()

    def forward(self, image_patch_features, text_patch_features, label):


        # --- Self-Attention before Fusion ---
        feature_img_sa, _ = self.self_attn_before_fusion(
            image_patch_features, image_patch_features, image_patch_features
        )  # (B, L_img, D)
        feature_txt_sa, _ = self.self_attn_before_fusion(
            text_patch_features, text_patch_features, text_patch_features
        )  # (B, L_txt, D)

        img2txt_patch = self.self_attn(
            feature_img_sa,  # Q: (B, L_img, D)
            feature_txt_sa,  # K: (B, L_txt, D)
            feature_txt_sa  # V
        )

        txt2img_patch = self.self_attn(
            feature_txt_sa,  # Q: (B, L_txt, D)
            feature_img_sa,  # K: (B, L_img, D)
            feature_img_sa
        )

        # --- Pooling ---
        w_img = torch.softmax(self.cross_pool(img2txt_patch), dim=1)
        feature_img_pooled = (w_img * img2txt_patch).sum(dim=1, keepdim=True)

        w_txt = torch.softmax(self.cross_pool(txt2img_patch), dim=1)
        feature_txt_pooled = (w_txt * txt2img_patch).sum(dim=1, keepdim=True)

        img_pooled = feature_img_sa.mean(dim=1, keepdim=True)
        txt_pooled = feature_txt_sa.mean(dim=1, keepdim=True)

        img, txt, ortho_loss, kl_div_loss, cross_entropy_loss = self.extraction(img_pooled, txt_pooled, label, eps=1e-6)

        alpha = torch.sigmoid(self.img2txt_weight)
        fusion_features = (1 - alpha) * feature_txt_pooled + alpha * feature_img_pooled

        gated_inputs = [
            self.pre_gated_norm(img),
            self.pre_gated_norm(txt),
            self.pre_gated_norm(feature_img_pooled),
            self.pre_gated_norm(feature_txt_pooled),
            self.pre_gated_norm(fusion_features),
        ]

        gated_list = self.gated_fusion(gated_inputs)

        #  (B, 5, D)
        fusion_3d = torch.cat(
            gated_list,
            dim=1
        )

        return fusion_3d, ortho_loss, kl_div_loss, cross_entropy_loss


class Hybrid_model(nn.Module):
    def __init__(self, embed_dim):
        super(Hybrid_model, self).__init__()
        self.embed_dim = embed_dim
        self.fusion = Fusion(embed_dim, num_classes=2)


        self.q_text_encoder = QwenEncoder(embed_dim)
        self.q_image_encoder = VitImageEncoder(embed_dim)

        self.loss = ClipInfoNCEloss(temperature=INFO_NCE_TEMPERATURE)

    def forward(self, img_q, text_q_ids, text_q_mask, labels):
        """
        Returns:
          total_contrastive_loss, loss_nce, loss_kl, itm_loss, fusion_logits, fused_output_2d, itm_prob
        """

        img_cls_features, img_patch_features = self.q_image_encoder(img_q)
        txt_cls_features, txt_patch_features = self.q_text_encoder(text_q_ids, text_q_mask)


        img_global_feat = torch.mean(img_cls_features, dim=1)
        txt_global_feat = torch.mean(txt_cls_features, dim=1)

        loss_cls = self.loss(img_global_feat, txt_global_feat, labels)


        fused_output_3d, ortho_loss, kl_div_loss, cross_entropy_loss = self.fusion(img_patch_features,
                                                                                   txt_patch_features, labels)

        total_contrastive_loss = loss_cls + ortho_loss + kl_div_loss + cross_entropy_loss
        # （B,3,D)
        return fused_output_3d, total_contrastive_loss



class Fusion3dProjection(nn.Module):
    """Qwen Projection"""

    def __init__(self, embed_dim, fusion_dim):
        super(Fusion3dProjection, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(embed_dim * 2, fusion_dim)
        )

    def forward(self, fused_features):
        return self.ffn(fused_features)


class QwenAdapter(nn.Module):
    """Generating Soft Prompt """

    def __init__(self, embed_dim):
        super(QwenAdapter, self).__init__()

        self.fusion_ffn = Fusion3dProjection(embed_dim, QWEN_HIDDEN_SIZE)
        self.norm = nn.RMSNorm(QWEN_HIDDEN_SIZE)
        self.dropout = nn.Dropout(0.4)

    def forward(self, fused_features):
        return self.dropout(self.norm(self.fusion_ffn(fused_features)))


class FocalLoss(nn.Module):
    def __init__(self, alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean',
                 label_smoothing=args.label_smoothing):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):

        pt = F.softmax(pred, dim=-1)
        pt_t = pt[torch.arange(pt.shape[0]), target]
        pt_t = (1 - self.label_smoothing) * pt_t + self.label_smoothing / 2

        # Label Weighting
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)  # (B,)

        loss = -alpha_t * ((1 - pt_t) ** self.gamma) * torch.log(pt_t + 1e-8)  # (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QwenWithAdapter(nn.Module):
    """
    Verbalizer：
    """

    def __init__(self, qwen_vl_model, adapter_embed_dim):
        super(QwenWithAdapter, self).__init__()
        self.qwen_vl_model = qwen_vl_model
        self.moco_model = Hybrid_model(adapter_embed_dim)
        self.adapter = QwenAdapter(adapter_embed_dim)

        self.tokenizer = AutoTokenizer.from_pretrained('./Qwen/Qwen3-VL-2B-Instruct', trust_remote_code=True)

        self.soft_prompt = """
        请严格按照以下要求判断图文内容是否为虚假新闻：
        1. 仅根据提供的图像和文本内容进行判断；
        2. 输出只能是“是”或“否”，无需额外解释；
        3. 虚假新闻输出“是”，真实新闻输出“否”。

        文本内容：{text}
        """

        self.token_yes_id = self.tokenizer.encode("是", add_special_tokens=False)[0]
        self.token_no_id = self.tokenizer.encode("否", add_special_tokens=False)[0]
        self.criterion = FocalLoss()


    def forward(
            self,
            input_ids: torch.Tensor,
            img_q,
            txt_q_mask: torch.Tensor,
            attention_mask: torch.Tensor,
            label: torch.Tensor = None,
    ):
        device = input_ids.device
        batch_size, text_len = input_ids.shape

        batch_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prompt_list = [self.soft_prompt.format(text=text) for text in batch_texts]
        tokenized_prompt = self.tokenizer(
            prompt_list,  # prompt
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        prompt_ids = tokenized_prompt["input_ids"].to(device)
        prompt_mask = tokenized_prompt["attention_mask"].to(device)
        Lp = prompt_ids.size(1)

        #  prefix
        fused_output_3d, total_contrastive_loss = self.moco_model(
            img_q, input_ids, txt_q_mask, label
        )
        prefix_embeds = self.adapter(fused_output_3d)  # (B, Np, D)
        Np = prefix_embeds.size(1)

        # Txt Embedding（Qwen3-VL）

        text_embeds = self.qwen_vl_model.language_model.embed_tokens(input_ids)
        prompt_embeds = self.qwen_vl_model.language_model.embed_tokens(prompt_ids)

        image_embeds = None

        # Concat prefix + image + text + prompt

        embeds_to_concat = [prefix_embeds]

        if image_embeds is not None:
            embeds_to_concat.append(image_embeds)

        embeds_to_concat.extend([text_embeds, prompt_embeds])

        combined_embeds = torch.cat(embeds_to_concat, dim=1)
        N_img = image_embeds.size(1) if image_embeds is not None else 0

        # Attention Mask
        prefix_mask = torch.ones((batch_size, Np), device=device)
        if image_embeds is not None:
            image_mask = torch.ones((batch_size, N_img), device=device)
        else:
            image_mask = torch.zeros((batch_size, 0), device=device)

        new_attention_mask = torch.cat([
            prefix_mask,
            image_mask,
            attention_mask,
            prompt_mask
        ], dim=1)

        # Position Ids
        prefix_pos = torch.zeros((batch_size, Np), dtype=torch.long, device=device)
        image_pos = torch.zeros((batch_size, N_img), dtype=torch.long, device=device)

        text_pos = torch.arange(text_len, device=device).unsqueeze(0).expand(batch_size, -1)
        text_padding_mask = (attention_mask == 0)

        text_pos = text_pos.masked_fill(text_padding_mask, 0)

        prompt_pos = torch.zeros((batch_size, Lp), dtype=torch.long, device=device)

        position_ids = torch.cat([prefix_pos, image_pos, text_pos, prompt_pos], dim=1)

        out = self.qwen_vl_model(
            inputs_embeds=combined_embeds,
            attention_mask=new_attention_mask,
            return_dict=True
        )

        logits_full = out.logits

        last_token_indices = new_attention_mask.sum(dim=1) - 1
        last_token_indices = last_token_indices.long()
        batch_indices = torch.arange(batch_size, device=device).long()
        last_token_logits = logits_full[batch_indices, last_token_indices]

        # verbalizer: 是 / 否

        score_no = last_token_logits[:, self.token_no_id]
        score_yes = last_token_logits[:, self.token_yes_id]

        target_logits = torch.stack([score_no, score_yes], dim=1)
        probs = F.softmax(target_logits, dim=-1)[:, 1]

        # Loss
        loss = None
        if label is not None:
            target = label.long().to(device)
            loss = self.criterion(target_logits, target)


        if loss is not None:
            total_loss = loss + CONTRAT_LOSS_WEIGHT * total_contrastive_loss
        else:
            total_loss = None

        return target_logits, total_loss


def calculate_metrics(all_labels: list, all_predictions: list) -> Dict[str, float]:

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_predictions)

    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    tn = np.sum((labels == 0) & (preds == 0))

    epsilon = 1e-6
    accuracy = (tp + tn) / (len(labels) + epsilon)

    # Rumor (Class 1) Metrics
    precision_rumor = tp / (tp + fp + epsilon)
    recall_rumor = tp / (tp + fn + epsilon)
    f1_rumor = 2 * (precision_rumor * recall_rumor) / (precision_rumor + recall_rumor + epsilon)

    # Non-Rumor (Class 0) Metrics
    precision_nonrumor = tn / (tn + fn + epsilon)
    recall_nonrumor = tn / (tn + fp + epsilon)
    f1_nonrumor = 2 * (precision_nonrumor * recall_nonrumor) / (precision_nonrumor + recall_nonrumor + epsilon)

    macro_f1 = (f1_rumor + f1_nonrumor) / 2

    return {
        'accuracy': accuracy,
        'f1_rumor': f1_rumor,
        'precision_rumor': precision_rumor,
        'recall_rumor': recall_rumor,
        'f1_nonrumor': f1_nonrumor,
        'precision_nonrumor': precision_nonrumor,
        'recall_nonrumor': recall_nonrumor,
        'macro_f1': macro_f1
    }


def evaluate_model(model_qwen_adapter: nn.Module, dataloader: DataLoader,
                   device: torch.device) -> Tuple[float, float, Dict[str, float]]:

    model_qwen_adapter.eval()
    total_loss_sum = 0.0
    num_batches = 0

    all_labels = []
    all_predictions = []

    test_bar = tqdm(dataloader, desc="Testing", leave=False)

    QWEN_DTYPE = model_qwen_adapter.qwen_vl_model.dtype

    with torch.no_grad():
        for batch in test_bar:
            if batch is None:
                continue

            img_q = batch['image_q']
            txt_q_ids = batch['input_ids'].to(device)
            txt_q_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # pixel_values = batch['pixel_values'].to(device)

            with torch.autocast(device_type=device.type, dtype=QWEN_DTYPE, enabled=device.type == 'cuda'):

                logits, loss = model_qwen_adapter(
                    input_ids=txt_q_ids,
                    img_q=img_q,
                    txt_q_mask=txt_q_mask,
                    attention_mask=txt_q_mask,
                    label=labels,
                    # pixel_values=pixel_values,
                )

            total_loss_sum += loss.item()
            num_batches += 1

            predictions = torch.argmax(logits, dim=-1)

            all_labels.append(labels.cpu().flatten().numpy())
            all_predictions.append(predictions.cpu().flatten().numpy())

            test_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss_sum / (num_batches if num_batches > 0 else 1)

    if len(all_labels) == 0 or len(all_predictions) == 0:
        print("Warning: No valid labels/predictions collected during evaluation.")
        metrics = {'accuracy': 0.0}
    else:
        metrics = calculate_metrics(all_labels, all_predictions)

    return avg_loss, metrics.get('accuracy', 0.0), metrics

if __name__ == '__main__':

    torch.manual_seed(32)
    QWEN_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"CUDA BF16 Support: {torch.cuda.is_bf16_supported()}")
    print(f"Qwen Model will be loaded and compute using: {QWEN_DTYPE}")

    scaler = torch.amp.GradScaler('cuda', enabled=(QWEN_DTYPE == torch.float16))
    MAX_GRAD_NORM = args.max_grad_norm

    qwen = AutoModelForCausalLM.from_pretrained(
        args.qwen_model_path,
        trust_remote_code=True,
        torch_dtype=QWEN_DTYPE,
        attn_implementation="sdpa"
    ).to(device)

    qwen.gradient_checkpointing_enable()


    for name, param in qwen.named_parameters():
        param.requires_grad = False


    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    qwen = get_peft_model(qwen, lora_config)

    # Qwen Adapter Model (Trainable)
    qwen_adapter_model = QwenWithAdapter(qwen, QWEN_HIDDEN_SIZE).to(device)
    qwen_adapter_model.to(QWEN_DTYPE)


    trainable_params = []
    moco_trainable_count = 0

    qwen_adapter_trainable_count = 0
    for name, param in qwen_adapter_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            qwen_adapter_trainable_count += param.numel()

    print(f"\n Total Trainable Parameters: {moco_trainable_count + qwen_adapter_trainable_count}")
    print(len(train_dataloader), len(test_dataloader))

    optimizer_grouped_parameters = [
        {
            'params': qwen_adapter_model.parameters(),
            'lr': args.lr,
            'name': 'heads_fusion'
        },
    ]


    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)


    TOTAL_STEPS = len(train_dataloader) * NUM_EPOCHS
    WARMUP_RATIO = args.warmup_ratio
    WARMUP_STEPS = int(TOTAL_STEPS * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=WARMUP_STEPS,
                                                num_training_steps=TOTAL_STEPS,
                                                num_cycles=0.5)

    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    patience_counter = 0
    best_test_accuracy = -1.0
    best_test_metrics = {}

    os.makedirs(os.path.dirname(NEW_MODEL_SAVE_PATH) or '.', exist_ok=True)


    train_correct_predictions = 0
    train_total_samples = 0
    global_step = 0
    ACCUMULATION_STEPS = args.accumulation_steps


    for epoch in range(NUM_EPOCHS):
        qwen_adapter_model.train()

        epoch_start_time = time.time()
        train_total_loss_sum = 0.0
        train_num_batches = 0

        optimizer.zero_grad()
        train_correct_predictions_epoch = 0
        train_total_samples_epoch = 0

        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} (Train)", leave=True)

        for batch_idx, batch in enumerate(train_bar):
            if batch is None:
                continue

            try:
                img_q = batch['image_q']
                txt_q_ids = batch['input_ids'].to(device)
                txt_q_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                # pixel_values = batch['pixel_values'].to(device)
            except KeyError:
                continue

            with torch.autocast(device_type=device.type, dtype=QWEN_DTYPE, enabled=device.type == 'cuda'):
                logits, loss = qwen_adapter_model(
                    input_ids=txt_q_ids,
                    txt_q_mask=txt_q_mask,
                    img_q=img_q,
                    attention_mask=txt_q_mask,
                    label=labels,
                    # pixel_values=pixel_values,
                )

            loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()


            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            predictions = torch.argmax(logits, dim=-1)
            batch_correct = (predictions == labels).sum().item()

            train_correct_predictions_epoch += batch_correct
            train_total_samples_epoch += labels.size(0)

            train_total_loss_sum += loss.item()
            train_num_batches += 1

            train_bar.set_postfix({
                'ls': f'{loss.item():.4f}',
                'Acc': f'{train_correct_predictions_epoch / train_total_samples_epoch:.4f}'
            })

        if (len(train_dataloader) % ACCUMULATION_STEPS) != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        avg_train_loss = train_total_loss_sum / (train_num_batches if train_num_batches > 0 else 1)
        test_loss, test_accuracy, test_metrics = evaluate_model(
            qwen_adapter_model, test_dataloader, device
        )

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        current_test_accuracy = test_accuracy
        epoch_end_time = time.time()

        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} Finished (Time: {epoch_end_time - epoch_start_time:.2f}s)")
        print(f"  [Train] Avg Loss: {avg_train_loss:.4f}")
        print(f"  [Train] Acc: {train_correct_predictions_epoch / train_total_samples_epoch:.4f}")
        print(f"  [Test] Avg Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.4f}")
        print(
            f"  [Metrics] Non-Rumor (0): P={test_metrics['precision_nonrumor']:.4f}, R={test_metrics['recall_nonrumor']:.4f}, F1={test_metrics['f1_nonrumor']:.4f}")
        print(
            f"  [Metrics] Rumor (1): P={test_metrics['precision_rumor']:.4f}, R={test_metrics['recall_rumor']:.4f}, F1={test_metrics['f1_rumor']:.4f} (Macro F1: {test_metrics['macro_f1']:.4f})")

        if current_test_accuracy > best_test_accuracy:
            best_test_accuracy = current_test_accuracy
            best_test_loss = test_loss
            best_test_metrics = test_metrics
            patience_counter = 0

            save_checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'test_loss': best_test_loss,
                'test_accuracy': best_test_accuracy,
                'test_metrics': best_test_metrics,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'qwen_adapter_state': qwen_adapter_model.adapter.state_dict(),
                'scaler_state': scaler.state_dict()
            }

            torch.save(save_checkpoint, NEW_MODEL_SAVE_PATH)
            print(f"\n>>>> Model Saved: New best Test Accuracy ({best_test_accuracy:.4f}) at epoch {epoch + 1}.")
        else:
            patience_counter += 1
            print(f"\nNo improvement in Test Accuracy. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("-" * 70)
                print(f"Early stopping triggered at epoch {epoch + 1}. Best Test Accuracy: {best_test_accuracy:.4f}")
                break