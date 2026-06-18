import sys
sys.path.append("/hpc2hdd/home/yxiao224/Henry/code/Uni3D")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    
import torch
import json
from typing import List, Dict, Any, Optional, Tuple

# ================= 配置区域 =================
MODEL_PATH = "/hpc2hdd/home/yxiao224/Henry/checkpoints/CLIP_eav02/open_clip_pytorch_model.bin"
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code2"
IMAGE_FEATURE_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/feature/"
IMAGE_FEATURE_ROOT1 = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/context_feature/"
IMAGE_FEATURE_ROOT2 = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/context_feature2/"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ================= 全局变量 (由外部 init_model 初始化) =================
device_global = None
clip_model_global = None
tokenizer_global = None

# 缓存字典 (仅保留图像缓存)
img_cache = {}

# ================= 核心初始化函数 (需手动调用) =================

def init_model(device_id: str = "1"):
    """
    手动初始化模型和设备。
    必须在调用 get_topk_candidates 之前执行一次。
    """
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
            device=device_global
        )
        tokenizer_global = SimpleTokenizer()
        print("   ✅ 模型初始化完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e

# ================= 辅助函数 =================

def _encode_text(identity_feat: str, category: str, des: Optional[str] = None) -> torch.Tensor:
    """
    内部文本编码函数。
    
    参数:
        identity_feat: 物体特征描述
        category: 物体类别
        des: 额外的约束描述 (红框中的物体应当符合下列描述...)。如果为 None，则只使用基础描述。
    """
    if clip_model_global is None:
        raise RuntimeError("模型未初始化！请先调用 init_model()。")

    # 如果没有提供 des，回退到旧逻辑
    if des is None or des.strip() == "":
        text_content = f"{identity_feat} {category}".strip()
        if not text_content:
            dim = clip_model_global.text_projection.out_features if hasattr(clip_model_global, 'text_projection') else 1280
            return torch.zeros(dim, device=device_global)
        
        prompts = f"a photo of a {category}, its feature is {identity_feat}."
    else:
        # 新增逻辑：使用 des 构建提示词
        prompts = f"The object in the red box should match the following description: {des}"#

    with torch.no_grad():
        tokens = tokenizer_global(prompts).to(device_global, non_blocking=True)
        if len(tokens.shape) < 2: 
            tokens = tokens[None, ...]
        
        emb = clip_model_global.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        avg = emb.mean(dim=0, keepdim=True)
        final_feat = avg / avg.norm(dim=-1, keepdim=True)
        return final_feat.squeeze(0)

# def _load_img_features_lazy(scene_id: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#     """
#     加载并缓存场景的图像特征。
#     同时从 IMAGE_FEATURE_ROOT 和 IMAGE_FEATURE_ROOT2 读取。
    
#     Returns:
#         Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: 
#             (root1_features, root2_features)
#     """
#     if scene_id in img_cache:
#         return img_cache[scene_id]
    
#     json_filename = f"{scene_id}_bbox.json"
    
#     def _load_single_path(root_path: str) -> Dict[str, torch.Tensor]:
#         local_feat_dict = {}
#         if not root_path:
#             return local_feat_dict
            
#         file_path = os.path.join(root_path, json_filename)
        
#         if os.path.exists(file_path):
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
                
#                 features_list = data.get("features", [])
#                 for item in features_list:
#                     obj_id = str(item.get("object_id"))
#                     feat_val = item.get("feature")
                    
#                     if isinstance(feat_val, list) and len(feat_val) > 0:
#                         local_feat_dict[obj_id] = torch.tensor(
#                             feat_val, dtype=torch.float32, device=device_global
#                         )
#             except Exception as e:
#                 print(f"Warning: Failed to load image features from {root_path} for {scene_id}: {e}")
        
#         return local_feat_dict

#     features_root1 = _load_single_path(IMAGE_FEATURE_ROOT)
#     features_root2 = _load_single_path(IMAGE_FEATURE_ROOT2)
    
#     result_tuple = (features_root1, features_root2)
#     img_cache[scene_id] = result_tuple
    
#     return features_root1, features_root2
def _load_img_features_lazy(scene_id: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    加载并缓存场景的图像特征。
    逻辑：优先读取 ROOT1；若成功，则 ROOT2 直接复用 ROOT1 的数据；
         若 ROOT1 失败，则尝试读取 ROOT2，若成功则两者均使用 ROOT2 的数据。
    
    Returns:
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: 
            (root1_features, root2_features)
            - 两种情况下，返回的两个字典内容完全一致（指向同一份有效数据或均为空）。
    """
    # 检查缓存
    if scene_id in img_cache:
        return img_cache[scene_id]
    
    json_filename = f"{scene_id}_bbox.json"
    
    def _load_single_path(root_path: str) -> Dict[str, torch.Tensor]:
        local_feat_dict = {}
        if not root_path:
            return local_feat_dict
            
        file_path = os.path.join(root_path, json_filename)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                features_list = data.get("features", [])
                for item in features_list:
                    obj_id = str(item.get("object_id"))
                    feat_val = item.get("feature")
                    
                    if isinstance(feat_val, list) and len(feat_val) > 0:
                        local_feat_dict[obj_id] = torch.tensor(
                            feat_val, dtype=torch.float32, device=device_global
                        )
            except Exception as e:
                print(f"Warning: Failed to load image features from {root_path} for {scene_id}: {e}")
        
        return local_feat_dict

    # 1. 尝试读取第一个路径 (IMAGE_FEATURE_ROOT)
    features_root_single = _load_single_path(IMAGE_FEATURE_ROOT)
    features_root1 = _load_single_path(IMAGE_FEATURE_ROOT1)
    features_root2 = _load_single_path(IMAGE_FEATURE_ROOT2)
    # 2. 判断逻辑
    # if features_root1:
    #     # 情况 A: 第一个路径读到了数据
    #     # -> 不用读第二个路径了
    #     # -> 两个返回值都使用这份数据
    #     pass
    #     # (可选) 打印日志确认使用了 fallback 逻辑
    #     # print(f"Info: Loaded features from ROOT1 for {scene_id}, skipping ROOT2.")
    # else:
    #     # 情况 B: 第一个路径没读到数据
    #     # -> 尝试读取第二个路径 (IMAGE_FEATURE_ROOT2)
    #     features_root1 = _load_single_path(IMAGE_FEATURE_ROOT2)


            
    # 3. 构建结果元组 (两个值相同)
    result_tuple = (features_root_single, features_root1,features_root2)
    
    # 4. 存入缓存
    img_cache[scene_id] = result_tuple
    
    return result_tuple

def _cosine_similarity(query_vec: torch.Tensor, db_vecs: torch.Tensor) -> torch.Tensor:
    """计算余弦相似度"""
    # query_vec: [D], db_vecs: [N, D]
    q_norm = query_vec / (query_vec.norm(dim=0, keepdim=True) + 1e-8)
    db_norms = db_vecs / (db_vecs.norm(dim=1, keepdim=True) + 1e-8)
    return torch.matmul(db_norms, q_norm)

# ================= 核心对外函数 =================

def get_topk_candidates(
    candidate_ids: List[str], 
    scene_id: str, 
    category: str, 
    identity_feat: str, 
    des: str,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    【双路融合模式】
    1. 使用 identity_feat 编码文本，与 IMAGE_FEATURE_ROOT (img_dict) 计算相似度。
    2. 使用 des ("红框中的物体...") 编码文本，与 IMAGE_FEATURE_ROOT2 (img_dict2) 计算相似度。
    3. 将两个相似度求平均进行排序。
    """
    if clip_model_global is None:
        raise RuntimeError("模型未初始化！请在调用此函数前先执行 init_model()。")
    
    # 1. 编码两个文本 Query
    try:
        # Query 1: 基于基础特征描述
        query_text_feat_1 = _encode_text(identity_feat, category, des=None)
        
        # Query 2: 基于红框约束描述
        query_text_feat_2 = _encode_text("", "", des=des) 
    except Exception as e:
        print(f"文本编码失败: {e}")
        return []

    # 2. 加载场景图像特征 (两个路径)
    img_dict, img_dict2 ,img_dict3 = _load_img_features_lazy(scene_id)
    
    if not img_dict or not img_dict2:
        print(f"⚠️ 警告：场景 {scene_id} 缺少部分图像特征 (Root1: {bool(img_dict)}, Root2: {bool(img_dict2)}).")
        # 如果任一路径完全为空，可能无法进行有效的双路融合，视情况返回空或降级处理
        # 这里选择如果任意一个为空则无法计算平均分，返回空
        return []

    # 3. 构建候选特征张量 (必须两个路径都有该 ID 才能参与双路评分)
    valid_ids = []
    img_feats_1 = []
    img_feats_2 = []

    for obj_id in candidate_ids:
        # 只有当 ID 在两个字典中都存在时，才加入计算列表
        if obj_id in img_dict:
            valid_ids.append(obj_id)
            img_feats_1.append(img_dict[obj_id])
            if obj_id in img_dict2:
                img_feats_2.append(img_dict2[obj_id])
            else:
                print("use new")
                img_feats_2.append(img_dict3[obj_id])
        else:
            # 可选策略：如果只想用单路补全，可以在这里处理，但目前逻辑是严格双路
            print("oh no")
            continue

    if not valid_ids:
        print(f"⚠️ 警告：提供的候选 ID 列表中没有找到同时在两个特征库中存在的对象。")
        return []

    # Stack 张量 [N, D]
    img_tensor_1 = torch.stack(img_feats_1)
    img_tensor_2 = torch.stack(img_feats_2)
    
    # 4. 分别计算相似度
    # Sims 1: Text(Identity) vs Image(Root1)
    sims_1 = _cosine_similarity(query_text_feat_1, img_tensor_1)
    
    # Sims 2: Text(Description) vs Image(Root2)
    sims_2 = _cosine_similarity(query_text_feat_2, img_tensor_2)
    
    # 5. 求平均作为最终分数
    # 确保两个相似度张量形状一致
    final_sims = (sims_1 + sims_2) / 2.0

    # 6. 获取 Top-K
    k = min(top_k, len(final_sims))
    if k == 0:
        return []
        
    topk_vals, topk_indices = torch.topk(final_sims, k=k)
    
    results = []
    for i, idx in enumerate(topk_indices.tolist()):
        results.append((valid_ids[idx], topk_vals[i].item()))
        
    return results

# ================= 调用样例 =================

if __name__ == "__main__":
    mock_candidate_ids = ["1", "5", "12", "23", "45", "88", "102"] 
    mock_scene_id = "birmingham_block_12"
    mock_category = "building"
    mock_identity_feat = "three large utility boxes beside it with large white grate tops"
    mock_des = "a tall structure with a flat roof and glass windows facing south" # 示例描述
    
    print(f"🚀 [测试模式] 开始对场景 {mock_scene_id} 进行双路融合重排序...")
    
    # 1. 初始化模型
    init_model(device_id="0") # 注意环境变量设的是 0
    
    # 2. 调用函数 (传入 des)
    topk_results = get_topk_candidates(
        candidate_ids=mock_candidate_ids,
        scene_id=mock_scene_id,
        category=mock_category,
        identity_feat=mock_identity_feat,
        des=mock_des,
        top_k=5
    )
    
    # 3. 打印结果
    print("\n" + "="*40)
    print(f"Top-5 检索结果 (Avg Similarity):")
    for rank, (obj_id, score) in enumerate(topk_results, 1):
        print(f"   Rank {rank}: ID = {obj_id}, Score = {score:.4f}")
    print("="*40)