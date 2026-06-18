import json
import os
import time
import random
import re
import base64
from PIL import Image
import io
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
BASE_DATA_PATH = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer"
META_FILE = os.path.join(BASE_DATA_PATH, "meta_data", "CityRefer_val_infer.jsonl")
META_FILE="/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/CityRefer_val_NO.json"
BOX3D_DIR = os.path.join(BASE_DATA_PATH, "box3d")
IMAGE_DIR = os.path.join(BASE_DATA_PATH, "redbox")
IMAGE_DIR2 = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/Context_image_new2"
# SGLang 配置
# SGLANG_BASE_URL = "http://127.0.0.1:8001/v1"
# MODEL_NAME = "Qwen/Qwen3.5-9B"
# API_KEY = "dummy-key"

SGLANG_BASE_URL = os.environ.get(
    "SGLANG_BASE_URL",
    os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = "qwen3.5-plus"


#在该任务中红框内的主体应当是￥￥￥
# 提示词

# PROMPT_TEMPLATE = """
# # Role
# 你是一位专业的遥感图像空间场景验证专家。你的任务是严格评估【图像描述】中的**周围环境信息**及**空间拓扑逻辑**是否与【红框标注区域】及其【周围环境】的视觉事实相符。

# # Critical Instructions (至关重要)
# 1. **主体预设**：红框内的主体物体已被确认为正确，**无需**评估主体本身的类别或属性。
# 2. **忽略绝对方位**：描述中可能包含“东、南、西、北”或“上、下、左、右”等绝对方位词。**由于遥感图像朝向未知，请完全忽略这些绝对方向指示**。
#    - 例如：若描述说“北侧有河”，你只需检查红框周围**是否有河**，而不必关心河是否在图像的“上方”。
#    - 只有当描述中的方位词隐含了**特定的拓扑结构**（如“两侧都有路”、“被...包围”）时，才验证该结构是否存在。
# 3. **核心关注点**：
#    - **有没有**：描述提到的环境客体（路、水、建筑等）在红框附近是否存在？
#    - **关系对不对**：主体与环境客体的**相对邻接关系**（紧邻、隔着、包围、之间）是否符合视觉事实？

# # Input Data
# - **图像内容**：一张包含红框标注的遥感图片。
# - **待评估描述**：{description}

# # Evaluation Guidelines (满分 100 分)

# ## 维度 1：周围客体信息存在性 (Context Object Existence) - 满分 50 分
# - **评估目标**：验证描述中提到的关键环境客体在红框可视范围内**是否真实存在**。
# - **判定规则**：
#   - 忽略描述中的方位词（如“北侧”、“左边”），只关注客体名词（如“河流”、“高速公路”、“学校”）。
#   - 只要该客体出现在红框附近的任何位置，即视为存在。
# - **评分标准**：
#   - **45-50 分**：描述中提到的所有关键环境客体均清晰可见。
#   - **30-44 分**：主要客体存在，但个别次要客体缺失或模糊。
#   - **10-29 分**：关键客体大部分缺失（如描述说有河，实际全是建筑）。
#   - **0-9 分**：描述中的环境客体完全不存在。

# ## 维度 2：与客体的相对拓扑关系 (Relative Topological Relation) - 满分 50 分
# - **评估目标**：验证主体与环境客体之间的**空间布局逻辑**是否相符。
# - **判定规则**：
#   - **忽略绝对方向**，重点验证**相对关系词**：
#     - “紧邻/挨着”：视觉上是否直接相连或距离极近？
#     - “隔着/中间有”：两者之间是否有描述中的阻隔物？
#     - “包围/环绕”：客体是否分布在主体的多个侧面？
#     - “之间”：主体是否位于两个客体中间？
#   - **矛盾判定**：如果描述说“紧邻河流”，但图像显示主体与河流之间隔了一条宽阔的道路或建筑群，则视为关系不匹配（即使河确实存在）。
# - **评分标准**：
#   - **45-50 分**：所有相对拓扑关系（邻接、阻隔、包围等）与图像高度一致。
#   - **30-44 分**：主要关系正确，但个别细节描述略有偏差（如“紧邻”实为“邻近但未接触”）。
#   - **10-29 分**：存在明显的拓扑矛盾（如描述说“被道路包围”，实际只有一侧有路）。
#   - **0-9 分**：空间布局逻辑完全错误。

# # Workflow
# 1. **提取客体**：从描述中提取环境客体列表，忽略方位词。
# 2. **验证存在性 (Score1)**：在图像中寻找这些客体，确认是否存在。
# 3. **验证拓扑关系 (Score2)**：分析主体与客体的视觉连接方式，判断“紧邻”、“隔着”、“包围”等逻辑是否成立。
# 4. **输出结果**：生成理由和分数。

# # Output Format
# 请严格按照以下步骤输出，不要输出任何多余的前缀或后缀：
# 1. 先输出一段简短的**空间环境分析理由**。
#    - 明确指出哪些客体存在。
#    - 明确指出哪些相对关系（如紧邻、隔绝）匹配或不匹配。
#    - **注意**：理由中不要纠结于“南北上下”是否正确，而要强调“客体是否存在”以及“相对位置逻辑是否成立”。
# 2. 换行后，输出一个标准的 JSON 对象：
#    - `reason`: 字符串，分析理由。
#    - `score1`: 整数，维度 1 (客体存在性) 得分 (0-50)。
#    - `score2`: 整数，维度 2 (相对拓扑关系) 得分 (0-50)。

# **示例输出**:
# 描述提到“北侧有河流”且“东侧紧邻高速公路”。图像中红框附近确实有一条河流和一条高速公路，客体均存在（维度 1 满分）。但在相对关系上，河流与红框之间隔了一片密集的住宅区和一条小路，并非描述中的“紧邻”或直接相邻关系；高速公路与红框之间也隔着一个停车场。因此拓扑关系存在偏差。
# {{"reason": "客体(河、高速)均存在，但均非紧邻关系，中间有明显阻隔物", "score1": 50, "score2": 30}}

# # Start Evaluation
# 描述：{description}
# """
PROMPT_TEMPLATE="""
# Role
你是一位专业的遥感图像与文本匹配评估专家。你的任务是评估给定的
【图像描述】与【红框标注区域】及其【周围环境】的匹配程度。

# Input Data
- **图像内容**：一张包含红框标注的遥感图片。
- **待评估描述**：{description}

# Evaluation Guidelines
请忽略背景中的嘈杂细节，也不要因为红框边缘的微小偏差而扣分（只要红框主要覆盖了目标物体即可），这个图片有可能没拍全整个场景。请重点关注以下两个维度：

## 维度 1：主体一致性 (Subject Consistency) - 满分 50 分
- **评估目标**：红框内的主体物体是否在**类别**和**关键属性**（如颜色、形状、结构、类型）上与描述完全一致。
在这里你要做的是着重关注主体的颜色与描述是否一致
- **评分标准**：
  - 45-50 分：主体类别完全正确，关键属性（如“红色屋顶”、“高层建筑”）完全匹配。
  - 30-44 分：主体类别正确，但部分非关键属性描述不准确或缺失。
  - 10-29 分：主体类别大致相关但有明显错误（如将主体的颜色描述错误）。
  - 0-9 分：主体完全错误或红框内无对应物体。
- **注意**：如果描述中提到特定的地名或路名（如“Hofford Drive”），请忽略这些特定命名信息，只关注视觉特征。

## 维度 2：空间拓扑关系 (Spatial Topology) - 满分 50 分
- **评估目标**：红框主体与周围相邻物体的**相对位置关系**是否与描述相符。
- **关键词关注**：关注描述中的方位词（如“北侧”、“旁边”、“紧邻”、“交叉处”、“被...包围”）。
- **评分标准**：
  - 45-50 分：所有提到的空间关系（相邻、方位、距离感）均与图像视觉事实高度一致。
  - 30-44 分：主要空间关系正确，但个别次要方位描述略有偏差。
  - 10-29 分：空间关系存在重大矛盾（如描述说“紧邻道路”，实际中间隔着建筑物）。
  - 0-9 分：空间关系完全错误或无法判断。

# Workflow
1. **观察分析**：首先仔细观察红框内的物体及其周围环境，对比描述中的每一个关键点。
2. **理由陈述**：用简练的语言列出匹配点和不匹配点。
3. **最终打分**：基于上述分析给出两个分数。

# Output Format
请严格按照以下步骤输出，不要输出任何多余的前缀或后缀：
1. 先输出一段简短的分析理由。
2. 换行后，输出一个标准的 JSON 对象，包含 `reason` (字符串), `score1` (整数), `score2` (整数)。

**示例输出**:
红框内确实是一栋高层住宅楼，颜色为灰色，与描述一致。但在空间关系上，描述称其“紧邻河流”，实际上该建筑与河流之间隔了一条主干道和一片绿地，空间关系不完全匹配。
{{"reason": "主体匹配但空间关系有误", "score1": 48, "score2": 25}}

# Start Evaluation
描述：{description}
"""
# ================= 功能函数 =================

def load_meta_data(file_path: str, limit: int = 100) -> List[Dict]:
    """
    加载元数据，自动支持 .json (列表) 和 .jsonl (逐行) 格式。
    
    Args:
        file_path: 文件路径
        limit: 最大读取条数
        
    Returns:
        包含字典的列表
    """
    data = []
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    with open(file_path, 'r', encoding='utf-8') as f:
        if ext == '.json':
            # 标准 JSON 格式：整个文件是一个列表 [...]
            full_data = json.load(f)
            if not isinstance(full_data, list):
                raise ValueError(f"JSON file {file_path} must contain a list at the root level.")
            data = full_data[:limit]
            
        elif ext in ['.jsonl', '.jsonlines']:
            # JSONL 格式：每一行是一个独立的 JSON 对象
            for line in f:
                if len(data) >= limit:
                    break
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line in {file_path}: {e}")
                    continue
        else:
            # 如果扩展名不明确，尝试通过读取第一行内容来启发式判断
            # 先读取所有行到列表中以便复用
            lines = f.readlines()
            if not lines:
                return []
            
            first_line = lines[0].strip()
            # 如果第一行以 '[' 开头且文件看起来像标准JSON (简单启发式)
            if first_line.startswith('['):
                # 重新组合内容并作为标准JSON加载
                # 注意：对于大文件，readlines() 可能消耗内存，但此处为了兼容性处理
                content = "".join(lines)
                full_data = json.loads(content)
                if not isinstance(full_data, list):
                    raise ValueError(f"File {file_path} looks like JSON but root is not a list.")
                data = full_data[:limit]
            else:
                # 默认按 JSONL 处理
                for line in lines:
                    if len(data) >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return data

RESULTS_JSON_PATH = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/newcode2/pipeline/evaluation_results_log.json"

def get_candidate_objects(
    scene_id: str, 
    target_obj_name: str, # 注：当前保存的JSON中未包含物体名称，此参数主要用于逻辑预留或外部校验，匹配主要靠ID
    gt_object_id: str, 
    ann_id: Any # 支持 int 或 str
) -> List[str]:
    """
    从评估结果日志中读取指定条目的成功 Top 30 候选列表。
    
    参数:
        scene_id: 场景ID
        target_obj_name: 目标物体名称 (当前逻辑中主要用于打印提示，匹配依赖ID)
        gt_object_id: 真实物体ID (Ground Truth ID)
        ann_id: 标注ID (annotation ID)
        
    返回:
        List[str]: 成功的 Top 30 候选物体ID列表。如果未找到或标记为失败，返回空列表。
    """
    
    if not os.path.exists(RESULTS_JSON_PATH):
        print(f"❌ 错误：结果文件不存在 -> {RESULTS_JSON_PATH}")
        print("   请先运行评估脚本生成该文件。")
        return []

    try:
        with open(RESULTS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 错误：读取 JSON 文件失败 -> {e}")
        return []

    # 标准化输入的 ann_id 为字符串进行比较，防止类型不匹配 (如 1 vs "1")
    target_ann_id_str = str(ann_id)
    target_gt_id_str = str(gt_object_id)
    target_scene_id_str = str(scene_id)

    for item in data:
        # 获取记录中的字段，并转换为字符串以确保比较安全
        rec_scene_id = str(item.get('scene_id', ''))
        rec_gt_id = str(item.get('object_id', ''))
        rec_ann_id = str(item.get('ann_id', ''))
        is_success = item.get('is_success', False)
        
        # 匹配逻辑：必须同时满足 scene_id, object_id, ann_id 一致
        if (rec_scene_id == target_scene_id_str and 
            rec_gt_id == target_gt_id_str and 
            rec_ann_id == target_ann_id_str):
            
            if is_success:
                candidates = item.get('candidates_20', [])
                # print(f"✅ 找到成功记录 (AnnID: {ann_id}), 返回 Top 30 候选列表 (共 {len(candidates)} 个)")
                return candidates
            else:
                # 找到了记录，但是 is_success 为 False
                # print(f"⚠️ 找到记录 (AnnID: {ann_id})，但该条目评估结果为失败 (is_success=False)。返回空列表。")
                return []

    # 如果遍历完都没找到
    # print(f"ℹ️ 未在日志中找到匹配的记录 (Scene: {scene_id}, GT: {gt_object_id}, AnnID: {ann_id})")
    return []

def get_image_path(scene_id: str, object_id: str) -> Optional[str]:
    """构造图片路径"""
    try:
        # 根据之前的逻辑，object_id 可能需要 +1 转换
        # 假设传入的 object_id 是字符串数字
        oid_int = int(object_id)
        display_id = str(oid_int)
    except ValueError:
        display_id = str(object_id)

    dir_path = os.path.join(IMAGE_DIR, scene_id)
    # 尝试格式 A: scene_id_obj{ID}.jpg
    filename_a = f"{scene_id}_obj{display_id}.jpg"
    full_path_a = os.path.join(dir_path, filename_a)
    
    if os.path.exists(full_path_a):
        return full_path_a
    # 尝试格式 B: scene_id_obj{ID}_.jpg (带下划线)
    filename_b = f"{scene_id}_{display_id}.png"
    full_path_b = os.path.join(IMAGE_DIR2, filename_b)
    print("补充")
    if os.path.exists(full_path_b):
        return full_path_b
        
    return None

# def convert_local_to_base64(image_path: str) -> str:
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')
def convert_local_to_base64(image_path: str, resize_factor: float = 0.5) -> str:
    """
    读取本地图片，可选缩放分辨率，然后转换为 Base64 字符串。
    
    Args:
        image_path: 图片本地路径
        resize_factor: 缩放比例 (0.5 表示长宽各减半，像素总量变为 1/4)
    """
    # 1. 打开图片
    with Image.open(image_path) as img:
        # 2. 如果需要缩放，则执行缩放
        if resize_factor != 1.0:
            original_width, original_height = img.size
            new_width = int(original_width * resize_factor)
            new_height = int(original_height * resize_factor)
            
            # 使用 LANCZOS 滤镜保证缩放质量
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 打印日志确认（可选）
            # print(f"Image resized from {original_width}x{original_height} to {new_width}x{new_height}")

        # 3. 将处理后的图片保存到内存缓冲区
        buffer = io.BytesIO()
        # 保持原格式保存，或者强制转为 JPEG/PNG 以进一步压缩
        # 如果原图是 PNG 且背景透明，建议保留 PNG；否则转 JPEG 可更小
        img_format = img.format if img.format else 'JPEG'
        img.save(buffer, format=img_format)
        
        # 4. 获取字节并编码为 Base64
        byte_data = buffer.getvalue()
        return base64.b64encode(byte_data).decode('utf-8')
import json
import json
import re

def parse_model_response_safe(raw_response):
    """
    安全解析模型响应，处理 content=None 和 混合文本+JSON 的情况
    """
    # 1. 基础空值检查
    if raw_response is None:
        print("⚠️ 错误：接收到的响应为 None")
        return "响应为空", 0, 0
    
    # 2. 确保输入是字符串 (防止传入 dict 或其他对象导致正则报错)
    if not isinstance(raw_response, str):
        print(f"⚠️ 错误：接收到的响应类型不是字符串，而是 {type(raw_response)}")
        # 尝试强制转换，如果不行则返回默认值
        try:
            raw_response = str(raw_response)
        except:
            return "类型转换失败", 0, 0

    # 3. 清理空白字符
    text_content = raw_response.strip()
    
    if not text_content:
        return "响应内容为空字符串", 0, 0

    # 4. 尝试提取 JSON
    # 策略：寻找最后一个 '{' 到最后一个 '}' 之间的内容，以应对文本中包含 '{' 的情况
    json_match = re.search(r'\{.*\}', text_content, re.DOTALL)
    
    extracted_json_str = ""
    reason_text = ""
    
    if json_match:
        extracted_json_str = json_match.group(0)
        # 提取 JSON 之前的文本作为理由 (如果没有 JSON 前的文本，就用 JSON 里的 reason)
        reason_text = text_content[:json_match.start()].strip()
    else:
        # 如果没有找到 JSON，尝试直接解析整个字符串（以防模型只发了 JSON）
        extracted_json_str = text_content
        reason_text = "未检测到详细理由，仅获取到分数"

    # 5. 解析 JSON 数据
    try:
        data = json.loads(extracted_json_str)
        
        # 兼容不同的 key 命名 (score1/score_1, reason/reasoning)
        score1 = int(data.get('score1', data.get('score_1', 0)))
        score2 = int(data.get('score2', data.get('score_2', 0)))
        
        # 优先使用 JSON 里的 reason，如果没有则使用提取的文本理由
        final_reason = data.get('reason', reason_text)
        
        if not final_reason:
            final_reason = "模型未提供具体理由"
            
        return final_reason, score1, score2

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败: {e}")
        print(f"   尝试解析的片段: {extracted_json_str[:100]}...")
        # 解析失败时的降级策略：返回 0 分或特定标记
        return f"JSON 格式错误：{str(e)}", 0, 0

# ==========================================
# 在你的主循环中这样调用 (假设 response 是 ChatCompletion 对象)
# ==========================================


def score_image_with_sglang(image_path: str, description: str) -> float:
    client = OpenAI(api_key=API_KEY, base_url=SGLANG_BASE_URL)

    try:
        base64_image = convert_local_to_base64(image_path)
        image_url_content = f"data:image/jpeg;base64,{base64_image}"
    except Exception as e:
        print(f"图片读取失败 {image_path}: {e}")
        return 0.0

    prompt_text = PROMPT_TEMPLATE.format(description=description)#.replace("￥￥￥",)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url_content}},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={"enable_thinking": True},
            #stream=True
            # max_tokens=32768, # 只需要一个分数，不需要太长
            # temperature=0.7, top_p=0.8, presence_penalty=1.5,
            # # extra_body={
            # #     "top_k": 20,
            # #     "chat_template_kwargs": {"enable_thinking": False},
            # # }
        )
        print(response)
        raw_text = response.choices[0].message.content.strip()
        print(f"Raw Response: {raw_text}")
        
        #numbers = re.findall(r"\d+", raw_text)
        numbers=0
        reason, s1, s2 = parse_model_response_safe(raw_text)
        numbers=s1+s2
        print(reason,s1,s2)
        if numbers:
            score = float(numbers)
            if score > 100: score = 100
            return score
        else:
            return 0.0
            
    except Exception as e:
        print(f"API 调用失败: {e}")
        return 0.0

def run_evaluation(num_samples: int = 100):
    num_samples=6000
    print(f"开始评估，目标样本数: {num_samples}")
    
    all_data = load_meta_data(META_FILE, limit=num_samples)
    if len(all_data) < num_samples:
        print(f"警告：数据集中只有 {len(all_data)} 条数据，将全部使用。")
        num_samples = len(all_data)

    total_correct = 0
    total_processed_samples = 0  # 分母：包含 GT 有图和 GT 无图的所有尝试过的样本
    total_time = 0.0
    results_log = []

    # 统计分类
    stats = {"gt_missing_image": 0, "normal": 0}
    step=1
    for idx, item in enumerate(tqdm(all_data[::step], desc="Evaluating")):
        scene_id = item["scene_id"]
        gt_object_id = str(item["object_id"])
        ann_id = str(item["ann_id"])
        object_name = item["object_name"]
        description = item["description"]
        # if idx<600 or item["landmark"]!=[]:
        #     continue
        #print()
        # 1. 获取候选列表 (尝试凑齐 10 个)
        # 注意：get_candidate_objects 现在会尝试用其他类物体填补空缺
        
        candidate_ids = get_candidate_objects(scene_id, object_name, gt_object_id,ann_id)
        total_processed_samples += 1
        # 2. 验证图片并分离 GT
        valid_candidates = [] # 存放 {id, path, is_gt}
        gt_path = None
        gt_exists = False
        if gt_object_id not in candidate_ids:
            stats["gt_missing_image"] += 1
            # GT 缺失，注定错误，但我们仍然可以跑一下模型看看它选谁（可选），或者直接记为错
            # 这里选择：依然跑模型，但最后强制 is_correct = False
            # 为了节省时间，如果 GT 缺失且你不想浪费推理资源，可以直接在这里记错并 continue
            # 但为了保持流程一致，我们继续往下走，最后判错。
            print("缺失")
            continue
        elif len(candidate_ids)==1:
            total_correct += 1
            stats["normal"] += 1
            continue
        
        for cid in candidate_ids:
            img_path = get_image_path(scene_id, cid)
            is_gt = (cid == gt_object_id)
            
            if img_path and os.path.exists(img_path):
                valid_candidates.append({"id": cid, "path": img_path, "is_gt": is_gt})
                if is_gt:
                    gt_exists = True
                    gt_path = img_path
            else:
                # 图片缺失
                if is_gt:
                    print(f"⚠️ GT 图片缺失: {scene_id} - {gt_object_id}")
                # 非 GT 缺失就直接丢弃，不加入列表

        # 【关键逻辑修改】
        
        if len(valid_candidates) < 2:
            # 连两个能看的图都没有，无法进行测试，跳过且不计数
            continue

        # 计入分母
        
        #print(f"实时准确率 (Top-1 Accuracy): ",(total_correct/total_processed_samples))


        # 3. 对每个有效图片打分
        start_time = time.time()
        scores = {}
        print(valid_candidates)
        for cand in valid_candidates:
            score = score_image_with_sglang(cand["path"], description)
            scores[cand["id"]] = score

        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        # 4. 找出最高分
        max_score = -1
        predicted_id = None
        
        for cid, sc in scores.items():
            if sc > max_score:
                max_score = sc
                predicted_id = cid
        # 获取 GT 的分数 (如果 GT 图片存在且在 scores 中)
        gt_score = scores.get(gt_object_id, None)
        
        # 格式化 GT 分数显示
        gt_score_str = f"{gt_score:.2f}" if gt_score is not None else "N/A (No Image)"
        
        # 打印详细信息
        print(f"🏆 最高分 ID: {predicted_id} ({max_score:.2f}) | 🎯 GT ID: {gt_object_id}_{scene_id} (Score: {gt_score_str})")
    
        # 5. 判断是否匹配 GT
        # 如果 GT 图片本身就不存在，predicted_id 永远不可能等于 gt_object_id (因为 gt_object_id 不在 scores 里)
        # 所以 is_correct 自然会是 False，符合逻辑
        is_correct = (predicted_id == gt_object_id)
        
        if is_correct :#or (int(max_score)==int(gt_score_str)):
            total_correct += 1
        
        # 实时准确率 (分母是 total_processed_samples)
        current_acc = total_correct / total_processed_samples if total_processed_samples > 0 else 0
        
        log_entry = {
            "scene_id": scene_id,
            "gt_id": gt_object_id,
            "gt_image_exists": gt_exists,
            "pred_id": predicted_id,
            "max_score": max_score,
            "is_correct": is_correct,
            "time": inference_time,
            "valid_count": len(valid_candidates),
            "all_scores": scores
        }
        results_log.append(log_entry)
        
        if (idx + 1) % 1 == 0:
            avg_time = total_time / total_processed_samples if total_processed_samples > 0 else 0
            print(f"\n[{total_processed_samples} 有效样本] 当前准确率: {current_acc:.2%}, 平均耗时: {avg_time:.2f}s/条")
            print(f"   (GT 缺失样本数：{stats['gt_missing_image']})")

    # 7. 最终统计
    final_accuracy = total_correct / total_processed_samples if total_processed_samples > 0 else 0
    avg_time_per_item = total_time / total_processed_samples if total_processed_samples > 0 else 0

    print("\n" + "="*30)
    print("评估完成")
    print(f"原始请求样本数: {num_samples}")
    print(f"实际有效测试条目数 (分母): {total_processed_samples}")
    print(f"   - GT 图片存在: {stats['normal']}")
    print(f"   - GT 图片缺失 (已计为错误): {stats['gt_missing_image']}")
    print(f"最终准确率 (Top-1 Accuracy): {final_accuracy:.4f} ({total_correct}/{total_processed_samples})")
    print(f"平均每条推理总耗时: {avg_time_per_item:.4f} 秒")
    print("="*30)

    with open("cityrefer_eval_results.json", "w", encoding='utf-8') as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print("详细结果已保存至 cityrefer_eval_results.json")

if __name__ == "__main__":
    random.seed(42)
    run_evaluation(num_samples=100)
