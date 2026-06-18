import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# =================配置区域=================
# 在这里粘贴你的完整 Prompt（使用 {description} 与 {地标列表} 占位符，勿使用 f-string，以免与 JSON 示例中的花括号冲突）
PROMPT_TEMPLATE = """
# Role
你是一个专业的文本信息提取助手。你的任务是从给定的文本描述中识别出一个**主要主体 (Main Subject)** 以及**潜在的多个客体对象 (Multiple Objects)**。

# Critical Constraints (必须严格遵守)
1.  **四大类限制 (Category Filter)**：
    *   主体和客体**必须**严格归类为以下四个预定义类别之一：`ground`, `building`, `car`, `parking`。。* 
    *   对于要寻找的主体，即使不明显属于这几类也要必须强行归类。对于客体，只能是这几种物体，不属于这几种的直接跳过。
    *   **过滤规则**：如果文中提到的物体（如 "trees", "bbox","running course", "people", "dogs", "lights"）等类似的明显不属于以上几类的，**无法**归入上述四类，**严禁**将其作为客体提取。直接忽略它们。

2.  **未知值处理**：
    *   对于任何无法从文本中提取的信息以及没有把握判断的信息，**必须**填入空字符串 `""`。

3.  **具体类别 (category2)**：提取文中提到的具体物体类别名称（例如："hotel", "school", "church", "sedan", "bus", "truck" 等）。如果文中没有明确提及具体类别，填入与大类一致的名称（如 "building"）；

4.  **颜色 (color)**：常见颜色即可，当有多种颜色的时候就写多种，但是当你不确定的时候只能为空。注意:如果没有直接提到房屋的颜色，那么屋顶就是房屋的颜色
5.  **属性特征 (identity_feature)**：
    *   仅描述固有属性（材质、形状等）。
    *   **禁止**包含空间关系（如 "next to", "in front of"）。颜色禁止在这里描述。
    *   若无私有属性描述，填 `""`。

6.  **地标 (landmark)**：
    *   地标只能输出我给你的地标列表中的选项，注意描述中的地标可能会跟选择列表中的有点区别。
    *   当提到一个地标时候，必须把其当作一个客体(比如看到The City Lawn就当作一个Ground类的客体)，当提到建筑物名字和街道名字,或者一个你不知道是什么的物体时，应该把其当作building客体。
    *   对于主体 (`main_object`)，一般不会提到主体的名字，此字段也保留，但填 `""`。
    *   对于客体，若不是地标物，Landmark字段填 `""`。

7.  **结构一致性 (Python Friendly)**：
    *   输出的所有对象（无论是主体还是客体列表中的项）必须拥有**完全相同的键**。
    *   键列表固定为：`id`, `category`, `category2`, `color`, `identity_feature`, `landmark`。
    *   `main_object` 的 `id` 固定为 `"main"`。
    *   `other_objects` 中的 `id` 依次为 `"object1"`, `"object2"`...

# Output JSON Structure
请严格按照以下 JSON 模板输出（不要输出 Markdown 标记，只输出纯 JSON 文本）：

{
  "main_object": {
    "id": "main",
    "category": "ground | building | car | parking",
    "category2": "",
    "color": "",
    "identity_feature": "",
    "landmark": ""
  },
  "other_objects": [
    {
      "id": "object1",
      "category": "ground | building | car | parking",
      "category2": "",
      "color": "",
      "identity_feature": "",
      "landmark": ""
    }
  ]
}

*注意：如果没有任何符合条件的客体，"other_objects" 必须是空数组 []。所有缺失信息均填 ""。*

# Examples

**输入 1：**
"This irregular green ground is on the intersection of London street and victoria street. The space between the road and the area is completely covered with trees."

**输出 1：**
{
  "main_object": {
    "id": "main",
    "category": "ground",
    "category2": "ground",
    "color": "green",
    "identity_feature": "irregular shape",
    "landmark": ""
  },
  "other_objects": [
    {
      "id": "object1",
      "category": "building",
      "category2": "street",
      "color": "",
      "identity_feature": "",
      "landmark": "victoria street"
    },
    {
      "id": "object2",
      "category": "building",
      "category2": "street",
      "color": "",
      "identity_feature": "",
      "landmark": "London street"
    }
  ]
}

**输入 2：**
"The large red brick hotel is located next to the Central House. A blue bus is parked in front of it near Fifth Avenue. There is a running course behind the hotel."

**输出 2：**
{
  "main_object": {
    "id": "main",
    "category": "building",
    "category2": "hotel",
    "color": "red",
    "identity_feature": "large, brick structure",
    "landmark": ""
  },
  "other_objects": [
    {
      "id": "object1",
      "category": "building",
      "category2": "House",
      "color": "",
      "identity_feature": "",
      "landmark": "Central House"
    },
    {
      "id": "object2",
      "category": "car",
      "category2": "bus",
      "color": "blue",
      "identity_feature": "",
      "landmark": ""
    },
    {
      "id": "object3",
      "category": "building",
      "category2": "Avenue",
      "color": "",
      "identity_feature": "",
      "landmark": "Fifth Avenue"
    },
  ]
}


# User Input
{description}{地标列表}
"""

DEFAULT_LANDMARK_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/landmark_all.json"

# 批量输入：多个 JSON 列表文件依次处理（与原先单文件逻辑相同，仅输出路径按输入推导）
INPUT_FILES = [
    "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_ND.json",
    "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO.json",
]


def input_json_to_output_jsonl(input_json: str) -> str:
    """与 CityRefer 一致：同目录下生成 {stem}_0324.jsonl。"""
    base, _ = os.path.splitext(input_json)
    return f"{base}_0324.jsonl"


def load_landmarks_by_scene(path: str) -> Dict[str, List[str]]:
    """从 landmark_all.json 读取：scene_id -> 地标名称列表。"""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v if x is not None and str(x).strip()]
        else:
            out[str(k)] = []
    return out


def format_landmark_block(scene_id: str, landmark_by_scene: Dict[str, List[str]]) -> str:
    """按场景拼接注入 Prompt 的地标列表段落。"""
    names = landmark_by_scene.get(str(scene_id), [])
    if not names:
        return (
            "\n\n## 本场景地标列表\n"
            f"场景 `{scene_id}` 无预定义地标清单；若文本中未出现可对应地标，landmark 填 \"\"。\n"
        )
    lines = "\n".join(f"- {n}" for n in names)
    return (
        "\n\n## 本场景可选地标列表（landmark 仅允许从下列选取或与之最接近的官方名称）\n"
        f"场景: `{scene_id}`\n"
        f"{lines}\n"
    )

# OpenAI Client Setup（DeepSeek 兼容 OpenAI SDK）
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

# DeepSeek Chat 模型名（与 dashscope 的 qwen 不同）
LLM_MODEL = "deepseek-chat"

def extract_json_content(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试从文本中提取合法的 JSON 对象。
    支持去除 Markdown 代码块标记。
    """
    if not text:
        return None
    
    # 1. 尝试去除 ```json ... ``` 包裹
    clean_text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s*```$', '', clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()
    
    # 2. 如果还是不行，尝试寻找第一个 { 和最后一个 }
    if not clean_text.startswith('{'):
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
        else:
            return None
            
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return None

def Infer(
    description: str,
    landmark_block: str = "",
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    调用 LLM 并解析结果。如果解析失败，自动重试。
    landmark_block 为按场景格式化的地标列表文本，注入 {地标列表}。
    """
    prompt_text = (
        PROMPT_TEMPLATE.replace("{description}", description).replace("{地标列表}", landmark_block)
    )
    
    messages = [
        {"role": "user", "content": prompt_text}
    ]

    for attempt in range(max_retries + 1):
        try:
            print(f"正在调用模型 (尝试 {attempt + 1}/{max_retries + 1})...")
            
            chat_response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=1024,
                top_p=0.8,
            )

            raw_text = chat_response.choices[0].message.content.strip()
            print(f"【原始输出片段】: {raw_text[:200]}...")

            parsed_data = extract_json_content(raw_text)
            
            if parsed_data and "main_object" in parsed_data:
                # 验证基本结构
                if "other_objects" not in parsed_data:
                    parsed_data["other_objects"] = []
                return parsed_data
            else:
                print(f"警告：第 {attempt + 1} 次尝试解析失败或结构不完整。")
                if attempt < max_retries:
                    time.sleep(1) # 短暂等待后重试
                    continue
                else:
                    print("错误：达到最大重试次数，仍无法解析有效 JSON。")
                    return None

        except Exception as e:
            print(f"调用 API 出错: {e}")
            if attempt < max_retries:
                time.sleep(1)
            else:
                return None

    return None


def load_existing_by_key(output_path: str) -> Dict[tuple, Dict[str, Any]]:
    """
    读取已存在的 JSONL 输出，按 (scene_id, object_id) 建立索引，用于断点续跑时跳过已处理行。
    """
    if not output_path or not os.path.exists(output_path):
        return {}
    existing: Dict[tuple, Dict[str, Any]] = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(obj.get("scene_id", "unknown"))
            oid = str(obj.get("object_id", "unknown"))
            existing[(sid, oid)] = obj
    return existing


def process_dataset(
    input_path: str,
    output_path: str,
    landmark_json_path: Optional[str] = None,
):
    # 1. 读取输入文件
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 {input_path}")
        return

    landmark_path = landmark_json_path if landmark_json_path is not None else DEFAULT_LANDMARK_JSON
    landmark_by_scene = load_landmarks_by_scene(landmark_path)
    if landmark_path and os.path.exists(landmark_path):
        print(f"已加载场景地标: {landmark_path}（共 {len(landmark_by_scene)} 个场景键）")
    else:
        print(f"警告：未找到地标文件 {landmark_path}，将不注入场景地标列表。")

    print(f"正在加载输入文件: {input_path} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    print(f"共加载 {len(all_data)} 条原始数据")

    existing_by_key = load_existing_by_key(output_path)
    if existing_by_key:
        print(f"检测到已有输出 {output_path}，共 {len(existing_by_key)} 条可复用记录；匹配到的行将跳过 API 调用。")

    # 2. 打开输出文件 (覆盖模式，按输入顺序整文件重写，已处理行从缓存写入)
    print(f"正在生成新文件: {output_path} (所有物体将合并到 construction 列表中)...")

    skipped_count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, item in enumerate(all_data):
            scene_id = str(item.get('scene_id', 'unknown'))
            original_object_id = str(item.get('object_id', 'unknown'))
            row_key = (scene_id, original_object_id)
            description = item.get('description', '')

            if row_key in existing_by_key:
                skipped_count += 1
                new_item = existing_by_key[row_key]
                if (i + 1) % 10 == 0 or i == len(all_data) - 1:
                    print(f"处理进度: {i+1}/{len(all_data)} [跳过已处理] | Scene: {scene_id}, Obj: {original_object_id}")
                try:
                    json.dump(new_item, f_out, ensure_ascii=False)
                    f_out.write('\n')
                except Exception as e:
                    print(f"写入错误: {e}")
                f_out.flush()
                continue

            if (i + 1) % 10 == 0 or i == len(all_data) - 1:
                print(f"处理进度: {i+1}/{len(all_data)} | Scene: {scene_id}, Obj: {original_object_id}")

            # --- 调用推理（按 scene_id 注入地标列表）---
            landmark_block = format_landmark_block(scene_id, landmark_by_scene)
            result = Infer(description, landmark_block)
            
            # 【核心修改】construction 现在是一个列表，用来存放该场景下所有的物体（主体+客体）
            construction_list = []

            # 辅助函数：构建单个物体的属性字典
            def build_object_props(obj_data: Dict) -> Dict:
                return {
                    "category": obj_data.get('category', ''),
                    "category2": obj_data.get('category2', ''),
                    "color": obj_data.get('color', ''),
                    "identity_feature": obj_data.get('identity_feature', ''),
                    "landmark": obj_data.get('landmark', ''),
                    # 可选：标记这是主体还是客体，方便下游区分
                    "is_main": obj_data.get('is_main', False) 
                }

            if result:
                # 1. 处理主体 (Main Object)
                main_obj = result.get('main_object', {})
                if main_obj:
                    # 标记为主体
                    main_props = build_object_props(main_obj)
                    main_props['is_main'] = True
                    construction_list.append(main_props)

                # 2. 处理客体 (Other Objects) - 直接追加到同一个列表中
                other_objs = result.get('other_objects', [])
                for idx, obj in enumerate(other_objs):
                    if not obj: 
                        continue
                    
                    # 标记为客体
                    sub_props = build_object_props(obj)
                    sub_props['is_main'] = False
                    # 可选：保留原始索引信息
                    sub_props['sub_index'] = idx 
                    
                    construction_list.append(sub_props)
            else:
                # 推理失败处理
                print(f"  [警告] Scene {scene_id} 推理失败，写入空 construction 列表。")
                # 失败时保持 construction 为空列表，或者放入一个空对象，视需求而定
                # 这里选择放入一个空对象占位，或者保持列表为空
                pass 

            # --- 构建最终要写入的条目 ---
            # 复制原始数据，确保不修改原始内存中的 item
            new_item = item.copy()
            
            # 【关键操作】将整个 construction 列表赋值给字段
            new_item['construction'] = construction_list
            
            # 注意：这里不再修改 object_id 或 ann_id，因为所有物体都在一个列表里了
            # 原始的 object_id 依然代表这条数据的主 ID
            
            # --- 写入文件 ---
            try:
                json.dump(new_item, f_out, ensure_ascii=False)
                f_out.write('\n')
            except Exception as e:
                print(f"写入错误: {e}")
            
            # 实时刷新
            f_out.flush()

    print(f"\n✅ 所有处理完成！结果已保存至: {output_path}")
    if skipped_count:
        print(f"本次跳过已处理行数: {skipped_count}（未重新调用 API）。")
    print("💡 数据结构说明：每条数据包含一个 'construction' 列表，列表内包含该场景识别出的所有物体（主体+客体）。")

if __name__ == "__main__":
    for idx, INPUT_FILE in enumerate(INPUT_FILES):
        OUTPUT_FILE = input_json_to_output_jsonl(INPUT_FILE)
        print(f"\n{'='*60}\n[{idx + 1}/{len(INPUT_FILES)}] 输入: {INPUT_FILE}\n输出: {OUTPUT_FILE}\n{'='*60}")
        process_dataset(INPUT_FILE, OUTPUT_FILE, landmark_json_path=DEFAULT_LANDMARK_JSON)
