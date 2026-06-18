import sys

sys.path.append("/hpc2hdd/home/yxiao224/Henry/code/Uni3D")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import torch
import json
from typing import List, Tuple, Optional

# 分轮淘汰：每轮去掉当前池中相似度最低的比例；字段为空则跳过该轮。
ELIMINATE_RATIO = 0.2

# ================= 配置区域 =================
MODEL_PATH = "/hpc2hdd/home/yxiao224/Henry/checkpoints/CLIP_eav02/open_clip_pytorch_model.bin"
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code2"
# 仅单图 bbox 特征（不使用 context / context2）
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
        tokenizer_global = SimpleTokenizer()
        print("   ✅ 模型初始化完成 (Indentity_in_stage1_new)")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e


def _encode_subject(category2: str, color: str, identity_feat: str) -> torch.Tensor:
    """
    单路文本编码：主体 construction 的 category2、color、identity_feature。
    """
    if clip_model_global is None:
        raise RuntimeError("模型未初始化！请先调用 init_model()。")

    c2 = (category2 or "").strip()
    col = (color or "").strip()
    ident = (identity_feat or "").strip()

    # 自然语言拼成一句，便于 CLIP
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

    with torch.no_grad():
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
    """在当前候选数下应淘汰的人数：至少 0，至多 pool_size - 1。"""
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


def get_topk_candidates(
    candidate_ids: List[str],
    scene_id: str,
    category2: str,
    color: str,
    identity_feat: str,
    top_k: int = 40,
) -> List[Tuple[str, float]]:
    """
    按「最多三轮、粗到细」用单字段文本与**单图**图像特征算相似度，每轮淘汰当前池中
    相似度最低的一批（比例 ELIMINATE_RATIO）；某字段为空则跳过该轮，不因此淘汰。

    不再用完整描述重排：返回顺序与分数来自「最后一次实际发生淘汰」的那一轮；
    若从未淘汰（字段全空或池子始终过小），则按原始候选顺序取前 top_k，分数为 0.0。
    """
    if clip_model_global is None:
        raise RuntimeError("模型未初始化！请在调用此函数前先执行 init_model()。")

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

    # 分轮：仅 category2 -> 仅 color -> 仅 identity（字段有值才参与淘汰）
    elimination_specs: List[Tuple[str, str, str, str]] = []
    if _field_nonempty(category2):
        elimination_specs.append(("category2", category2, "", ""))
    if _field_nonempty(color):
        elimination_specs.append(("color", "", color, ""))
    if _field_nonempty(identity_feat):
        elimination_specs.append(("identity", "", "", identity_feat))

    active_ids = list(valid_ids)
    last_scores: Optional[List[float]] = None

    for _name, c2, col, ident in elimination_specs:
        try:
            q_round = _encode_subject(c2, col, ident)
        except Exception as e:
            print(f"分轮文本编码失败 ({_name}): {e}")
            continue
        n = len(active_ids)
        n_drop = _eliminate_count(n)
        if n_drop <= 0:
            continue
        row_idx = [id_to_row[oid] for oid in active_ids]
        sims = _single_img_sim_for_indices(q_round, img_tensor, row_idx)
        order = torch.argsort(sims, descending=True).tolist()
        keep_positions = order[: (n - n_drop)]
        active_ids = [active_ids[j] for j in keep_positions]
        last_scores = [float(sims[j].item()) for j in keep_positions]

    k = min(top_k, len(active_ids))
    if k == 0:
        return []

    results: List[Tuple[str, float]] = []
    for i in range(k):
        oid = active_ids[i]
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

    print(f"🚀 [测试] Indentity_in_stage1_new — {mock_scene_id}")
    init_model(device_id="0")
    topk_results = get_topk_candidates(
        candidate_ids=mock_candidate_ids,
        scene_id=mock_scene_id,
        category2=mock_c2,
        color=mock_color,
        identity_feat=mock_ident,
        top_k=5,
    )
    print("Top-5:", topk_results)
