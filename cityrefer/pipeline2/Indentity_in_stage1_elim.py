import sys

sys.path.append("/hpc2hdd/home/yxiao224/Henry/code/Uni3D")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import torch
import json
from typing import List, Tuple, Optional

# 串联末位淘汰：顺序为 类别 → 颜色 → 特征；每一轮只在**上一轮幸存池**上淘汰；
# 每轮文本只启用**当前这一维**（另两维传空串），故 prompt 走 _encode_subject 的「仅 c2 / 仅 col / 仅 ident」分支，而非三句拼成一句。
ELIMINATE_RATIO = 0.2

# ================= 配置区域 =================
MODEL_PATH = "/hpc2hdd/home/yxiao224/Henry/checkpoints/CLIP_eav02/open_clip_pytorch_model.bin"
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code2"
# 仅单图 bbox 特征（与 context / context2 无关）
IMAGE_FEATURE_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/feature/"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

device_global = None
clip_model_global = None
tokenizer_global = None
img_cache = {}


def init_model(device_id: str = "1"):
    global device_global, clip_model_global, tokenizer_global

    if clip_model_global is not None:
        print("✅ 模型已初始化，跳过。")
        return

    print(f"⏳ 正在初始化 CLIP 模型 (Device: {device_id})...")
    device_global = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import open_clip
        from utils.tokenizer import SimpleTokenizer

        clip_model_global, _ = open_clip.create_model_from_pretrained(
            model_name="EVA02-E-14-plus",
            pretrained=MODEL_PATH,
            device=device_global,
        )
        clip_model_global.eval()
        tokenizer_global = SimpleTokenizer()
        print("   ✅ 模型初始化完成 (Indentity_in_stage1_elim)")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e


def _encode_subject(category2: str, color: str, identity_feat: str) -> torch.Tensor:
    """
    淘汰流程里每轮只应传**一个**非空字段（另两个为 ''），对应下面「仅类别 / 仅颜色 / 仅特征」的短 prompt。
    若误传多字段非空，会走组合句分支（本模块的串联轮次不会如此调用）。
    """
    if clip_model_global is None:
        raise RuntimeError("模型未初始化！请先调用 init_model()。")

    c2 = (category2 or "").strip()
    col = (color or "").strip()
    ident = (identity_feat or "").strip()

    if c2 and col and ident:
        prompts = (
            f"a photo of a {c2}; its color is {col}; its distinguishing features are {ident}."
        )
    elif c2 and col:
        prompts = f"a photo of a {c2} in {col} color."
    elif c2 and ident:
        prompts = f"a photo of a {c2}; its distinguishing features are {ident}."
    elif c2:
        prompts = f"a photo of a {c2}."
    elif col and ident:
        prompts = f"a photo of an object in {col} color; its distinguishing features are {ident}."
    elif ident:
        prompts = f"a photo of an object; its distinguishing features are {ident}."
    elif col:
        prompts = f"a photo of an object in {col} color."
    else:
        dim = (
            clip_model_global.text_projection.out_features
            if hasattr(clip_model_global, "text_projection")
            else 1280
        )
        return torch.zeros(dim, device=device_global)

    with torch.inference_mode():
        tokens = tokenizer_global(prompts).to(device_global, non_blocking=True)
        if len(tokens.shape) < 2:
            tokens = tokens[None, ...]
        emb = clip_model_global.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        avg = emb.mean(dim=0, keepdim=True)
        final_feat = avg / avg.norm(dim=-1, keepdim=True)
        return final_feat.squeeze(0)


def _load_img_features_lazy(scene_id: str) -> dict:
    if scene_id in img_cache:
        return img_cache[scene_id]

    json_filename = f"{scene_id}_bbox.json"
    local_feat_dict: dict = {}
    file_path = os.path.join(IMAGE_FEATURE_ROOT, json_filename)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("features", []):
                obj_id = str(item.get("object_id"))
                feat_val = item.get("feature")
                if isinstance(feat_val, list) and len(feat_val) > 0:
                    local_feat_dict[obj_id] = torch.tensor(
                        feat_val, dtype=torch.float32, device=device_global
                    )
        except Exception as e:
            print(f"Warning: Failed to load image features from {IMAGE_FEATURE_ROOT} for {scene_id}: {e}")

    img_cache[scene_id] = local_feat_dict
    return local_feat_dict


def _cosine_similarity(query_vec: torch.Tensor, db_vecs: torch.Tensor) -> torch.Tensor:
    q_norm = query_vec / (query_vec.norm(dim=0, keepdim=True) + 1e-8)
    db_norms = db_vecs / (db_vecs.norm(dim=1, keepdim=True) + 1e-8)
    return torch.matmul(db_norms, q_norm)


def _field_nonempty(s: Optional[str]) -> bool:
    return bool((s or "").strip())


def _eliminate_count(pool_size: int, ratio: float = ELIMINATE_RATIO) -> int:
    if pool_size <= 1:
        return 0
    k = int(math.floor(pool_size * ratio))
    if k < 1:
        k = 1
    return min(k, pool_size - 1)


def _single_img_sim_for_indices(
    query_vec: torch.Tensor,
    img_tensor: torch.Tensor,
    indices: List[int],
) -> torch.Tensor:
    if not indices:
        return torch.empty(0, device=query_vec.device)
    idx_t = torch.tensor(indices, dtype=torch.long, device=query_vec.device)
    vecs = img_tensor.index_select(0, idx_t)
    return _cosine_similarity(query_vec, vecs)


def _eliminate_one_round(
    active_ids: List[str],
    id_to_row: dict,
    img_tensor: torch.Tensor,
    q_round: torch.Tensor,
) -> Tuple[List[str], Optional[List[float]]]:
    """
    在当前池上做一次末位淘汰。
    若无需淘汰（n_drop==0），返回 (active_ids, None)，表示与原版 `continue` 一致：不更新 last_scores。
    """
    n = len(active_ids)
    n_drop = _eliminate_count(n)
    if n_drop <= 0:
        return list(active_ids), None
    row_idx = [id_to_row[oid] for oid in active_ids]
    sims = _single_img_sim_for_indices(q_round, img_tensor, row_idx)
    k_keep = n - n_drop
    order = torch.argsort(sims, descending=True).tolist()[:k_keep]
    new_ids = [active_ids[j] for j in order]
    new_scores = [float(sims[j].item()) for j in order]
    return new_ids, new_scores


def get_eliminated_candidates_pair(
    candidate_ids: List[str],
    scene_id: str,
    category2: str,
    color: str,
    identity_feat: str,
    stop_elimination_when_below: int,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    一次串联仿真同时得到两种策略的结果（文本编码与各轮相似度在两队池一致时只算一遍）：
    - 「停」：与 get_eliminated_candidates(..., stop_elimination_when_below=stop) 一致；
    - 「满」：与 get_eliminated_candidates(..., stop_elimination_when_below=None) 一致。

    用于评估脚本，避免对同一条样本跑两次完整 CLIP。
    """
    if clip_model_global is None:
        raise RuntimeError("模型未初始化！请先调用 init_model()。")

    img_dict = _load_img_features_lazy(scene_id)
    if not img_dict:
        print(f"⚠️ 警告：场景 {scene_id} 缺少单图特征 ({IMAGE_FEATURE_ROOT})。")
        return [], []

    valid_ids: List[str] = []
    img_feats: List[torch.Tensor] = []
    for obj_id in candidate_ids:
        if obj_id not in img_dict:
            continue
        valid_ids.append(obj_id)
        img_feats.append(img_dict[obj_id])

    if not valid_ids:
        print("⚠️ 警告：候选 ID 在图像特征中均不可用。")
        return [], []

    img_tensor = torch.stack(img_feats)
    id_to_row = {oid: i for i, oid in enumerate(valid_ids)}

    elimination_specs: List[Tuple[str, str, str, str]] = []
    if _field_nonempty(category2):
        elimination_specs.append(("category2", category2, "", ""))
    if _field_nonempty(color):
        elimination_specs.append(("color", "", color, ""))
    if _field_nonempty(identity_feat):
        elimination_specs.append(("identity", "", "", identity_feat))

    if not elimination_specs:
        z = [(oid, 0.0) for oid in valid_ids]
        return z, z

    active_stop = list(valid_ids)
    active_full = list(valid_ids)
    last_scores_stop: Optional[List[float]] = None
    last_scores_full: Optional[List[float]] = None
    frozen_stop = False

    with torch.inference_mode():
        for _name, c2, col, ident in elimination_specs:
            try:
                q_round = _encode_subject(c2, col, ident)
            except Exception as e:
                print(f"分轮文本编码失败 ({_name}): {e}")
                continue

            if not frozen_stop and len(active_stop) < stop_elimination_when_below:
                frozen_stop = True

            run_stop = not frozen_stop
            run_full = True

            if run_stop and run_full and active_stop == active_full:
                new_ids, new_scores = _eliminate_one_round(
                    active_stop, id_to_row, img_tensor, q_round
                )
                active_stop = active_full = new_ids
                if new_scores is not None:
                    last_scores_stop = last_scores_full = new_scores
            else:
                if run_stop:
                    new_s, sc_s = _eliminate_one_round(
                        active_stop, id_to_row, img_tensor, q_round
                    )
                    active_stop = new_s
                    if sc_s is not None:
                        last_scores_stop = sc_s
                if run_full:
                    new_f, sc_f = _eliminate_one_round(
                        active_full, id_to_row, img_tensor, q_round
                    )
                    active_full = new_f
                    if sc_f is not None:
                        last_scores_full = sc_f

    def _pack(
        ids: List[str], scores: Optional[List[float]]
    ) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for i, oid in enumerate(ids):
            if scores is not None and i < len(scores):
                out.append((oid, scores[i]))
            else:
                out.append((oid, 0.0))
        return out

    return _pack(active_stop, last_scores_stop), _pack(active_full, last_scores_full)


def get_eliminated_candidates(
    candidate_ids: List[str],
    scene_id: str,
    category2: str,
    color: str,
    identity_feat: str,
    *,
    stop_elimination_when_below: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    **串联**末位淘汰：顺序为 category2 → color → identity；字段为空则跳过该维（不淘汰）。
    每一维执行时，仅用该维文本与单图特征算相似度（`_encode_subject` 当轮只带一个非空字段），
    在**当前幸存池**上去掉末尾比例 ELIMINATE_RATIO；下一轮在**缩小后的池**上继续。

    相似度仅基于单图 bbox 特征（IMAGE_FEATURE_ROOT）。返回全部幸存候选（好→差），不做 top-k。
    分数来自**最后一次实际发生淘汰**的那一轮；若从未淘汰则顺序为原始 valid 顺序、分数 0.0。

    stop_elimination_when_below:
        若设置（如 20），在进入下一轮淘汰**之前**，若当前池人数 < 该值则停止后续删减。
    """
    if clip_model_global is None:
        raise RuntimeError("模型未初始化！请先调用 init_model()。")

    img_dict = _load_img_features_lazy(scene_id)

    if not img_dict:
        print(f"⚠️ 警告：场景 {scene_id} 缺少单图特征 ({IMAGE_FEATURE_ROOT})。")
        return []

    valid_ids = []
    img_feats = []

    for obj_id in candidate_ids:
        if obj_id not in img_dict:
            continue
        valid_ids.append(obj_id)
        img_feats.append(img_dict[obj_id])

    if not valid_ids:
        print("⚠️ 警告：候选 ID 在图像特征中均不可用。")
        return []

    img_tensor = torch.stack(img_feats)
    id_to_row = {oid: i for i, oid in enumerate(valid_ids)}

    elimination_specs: List[Tuple[str, str, str, str]] = []
    if _field_nonempty(category2):
        elimination_specs.append(("category2", category2, "", ""))
    if _field_nonempty(color):
        elimination_specs.append(("color", "", color, ""))
    if _field_nonempty(identity_feat):
        elimination_specs.append(("identity", "", "", identity_feat))

    # 每轮 (c2,col,ident) 仅一个非空 → 单字段 prompt，串联缩小 active_ids
    active_ids = list(valid_ids)
    last_scores: Optional[List[float]] = None

    with torch.inference_mode():
        for _name, c2, col, ident in elimination_specs:
            if stop_elimination_when_below is not None and len(active_ids) < stop_elimination_when_below:
                break
            try:
                q_round = _encode_subject(c2, col, ident)
            except Exception as e:
                print(f"分轮文本编码失败 ({_name}): {e}")
                continue
            active_ids, new_scores = _eliminate_one_round(
                active_ids, id_to_row, img_tensor, q_round
            )
            if new_scores is not None:
                last_scores = new_scores

    results: List[Tuple[str, float]] = []
    for i, oid in enumerate(active_ids):
        if last_scores is not None:
            results.append((oid, last_scores[i]))
        else:
            results.append((oid, 0.0))
    return results


if __name__ == "__main__":
    mock_candidate_ids = ["1", "5", "12", "23", "45", "88", "102"]
    mock_scene_id = "birmingham_block_12"
    mock_c2 = "building"
    mock_color = "gray"
    mock_ident = "three large utility boxes beside it with large white grate tops"

    print(f"🚀 [测试] Indentity_in_stage1_elim — {mock_scene_id}")
    init_model(device_id="0")
    out = get_eliminated_candidates(
        candidate_ids=mock_candidate_ids,
        scene_id=mock_scene_id,
        category2=mock_c2,
        color=mock_color,
        identity_feat=mock_ident,
    )
    print(f"幸存者数量: {len(out)}")
    print("前 5 条:", out[:5])
