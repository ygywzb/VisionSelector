import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None
logger = logging.get_logger(__name__)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from copy import deepcopy
from torch import vmap
from torch.func import grad
from torch.autograd import Function

# --- Start of new Differentiable TopK implementation ---
sigmoid = torch.sigmoid
sigmoid_grad = vmap(vmap(grad(sigmoid)))

# Function是torch里的，因为这里需要定义反向传播
# 二分求t无需反向传播，因此需手写反向传播逻辑
class TopK(Function):
    @staticmethod
    def forward(ctx, xs, k):
        ts, ps = _find_ts(xs, k)
        ctx.save_for_backward(xs, ts)
        return ps

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        xs, ts = ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        v = sigmoid_grad(xs + ts)
        s = v.sum(dim=1, keepdims=True)
        # Jacobian is -vv.T/s + diag(v)
        uv = grad_output * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        return t1 + uv, None

@torch.no_grad()
def _find_ts(xs, k):
    b, n = xs.shape
    assert 0 < k < n
    # Lo should be small enough that all sigmoids are in the 0 area.
    # Similarly Hi is large enough that all are in their 1 area.
    lo = -xs.max(dim=1, keepdims=True).values - 10
    hi = -xs.min(dim=1, keepdims=True).values + 10
    for _ in range(64):
        mid = (hi + lo)/2
        mask = sigmoid(xs + mid).sum(dim=1) < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
    ts = (lo + hi)/2
    return ts, sigmoid(xs + ts)

# 转成和nn.module一样的调用方式
topk = TopK.apply
# --- End of new Differentiable TopK implementation ---

def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(q.dtype).chunk(2, dim=-1)[0].contiguous()
    sin = sin.to(q.dtype).chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q, cos, sin).type_as(q)
    k_embed = apply_rotary_emb(k, cos, sin).type_as(k)
    return q_embed, k_embed


def qwen25vl_vision_tower_forward_selector(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # ---------------------------add---------------------------------------------
    hidden_states_unsqueezed = hidden_states.unsqueeze(0)
    # 已经注入，就是打分器
    learned_scores = self.importance_scorer(hidden_states_unsqueezed).squeeze(0)
    total_tokens = learned_scores.shape[0]
    # 保留token的数量，k个
    k = int(total_tokens * self.budgets)
    img_mask = topk(learned_scores.unsqueeze(0), k).squeeze(0)
    # 拓展到hidden_dim
    img_mask_expanded = img_mask.unsqueeze(1).expand(-1, hidden_states_unsqueezed.shape[-1])
    hidden_states_new = img_mask_expanded*hidden_states
    hidden_states_new = hidden_states_new.type(hidden_states.dtype)

    with torch.no_grad():
        constraint_topk_indices = learned_scores.topk(k, dim=0).indices
        constraint_img_mask = torch.zeros_like(learned_scores, device=learned_scores.device)
        # scatter_：按照索引位置填充指定的值
        # 这里是找出前k个对应的位置填充为1
        constraint_img_mask.scatter_(dim=-1, index=constraint_topk_indices, value=1.0)
    # -------------------------------------------------------------------------------
    # 返回：可微topk后的hidden_states，软掩膜，硬掩膜
    return hidden_states_new, img_mask, constraint_img_mask


# mllm总的forward，会用他会用视觉编码器的forward
# 原本qwen模型的forward改成了这个
def qwen25vl_generation_forward_selector(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            # 这里的visual已经用创新代码替换了
            image_embeds, img_mask, constraint_img_mask = self.visual(pixel_values, grid_thw=image_grid_thw)
            # 检查视觉特征数量是否与文本中的图像占位符数量一致
            # image_token_id固定，和input_ids计算布尔索引后求和
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            # 将input_embeds中对应图像token的位置替换为视觉特征
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            # 和上面图像的一样
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # 视觉特征已经填入了inputs_embeds里了，可调用model的forward
    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position
    )

    hidden_states = outputs[0]
    # 预测的结果，最后一个维度是词表大小（没归一化的分布）
    # 每一个预测位置，都是结合其前面所有内容的结果，计算自注意力时会屏蔽token位置后面的所有内容
    logits = self.lm_head(hidden_states)
    

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        # 错位，logits去掉最后一个，labels去掉第一个
        # 对齐后，logict是参考了该位置及之前所有内容的预测分布，用来预测下一个位置的label
        # 让llm学到看到前几个字后，可以猜出下一个字是什么
        #                                   ——也就是学到语言统计规律、语法、语义、世界知识
        # 另外，这里喂的数据是图像理解的数据，有图有文字（图已经和文字对齐了），llm也会学会结合图像内容来预测文字
        # 若llm已经预训练，llm已经基本学会了语法这种宏观的知识，在此基础上学会结合图像内容来预测文字效果更好
        shift_logits = logits[..., :-1, :].contiguous()
        # 注意：错位后的labels删掉了bos，但没删掉eos
        # 使得llm可以在前面的序列是什么样的情况下要输出eos
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        # 定长seq_len的好处是可以直接拉平
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # 交叉熵：一个位置的预测分布和真实分布之间的距离，多个的话就是求平均
        #                                                       ——也就是衡量预测效果
        loss = loss_fct(shift_logits, shift_labels)
        # print('ce loss: {:.5f}'.format(loss.item()))

        # 以上是下游任务
        # --------------------------------add---------------------------------------------
        # 这里让llm额外学会在视觉特征不全的时候也能保持下游任务的效果（这里是单纯的llm推测）
        if pixel_values is not None:
            # 二值交叉熵，预测的软掩膜和硬掩膜之间的距离
            constraint_loss = F.binary_cross_entropy(img_mask, constraint_img_mask)
            # 课程退火策略的动态权重，在每次执行这个forward之前计算好
            loss += self.regularization_weight * constraint_loss
            # print('soft regularization_loss: {:.5f}'.format(self.regularization_weight * constraint_loss.item()))
        # -------------------------------------------------------------------------------

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )