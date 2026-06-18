import json
import os
import re
import time
from typing import Dict, Any, Optional
from multiprocessing import Pool
from functools import partial

# =================配置区域=================
# 在这里粘贴你的完整 Prompt
PROMPT_TEMPLATE = """
# Role
你是一个专业的空间信息提取助手。你的任务是从给定的文本描述中，识别出**主要主体 (Main Subject)** 和**客体对象 (Multiple Objects)**，并**原样摘抄**文本中连接它们的空间位置词汇。

# Critical Constraints (必须严格遵守)

1. **类别限制 (Category Filter)**：
   * 主体和客体**必须**严格归类为以下预定义类别之一：
     `ground`, `building`, `parking`, `vehicle`, `truck`, `highvegetation`, `fence`, `lightpole`
   * 主体必须强行归入上述类别之一。
   * 客体如果不属于上述类别，允许保留在 `reference_anchor` 原文中，但**不要**把它错误映射为别的类别。
   * 类别映射规则：
     * `car`, `cars`, `van`, `bike`, `bicycle`, `motorcycle` -> `vehicle`
     * `truck`, `trucks` -> `truck`
     * `tree`, `trees`, `bush`, `hedge`, `vegetation` -> `highvegetation`
     * `wall`, `walls`, `fence`, `gate` -> `fence`
     * `street lamp`, `streetlight`, `lamp post`, `light pole`, `traffic light` -> `lightpole`
     * `road`, `street`, `intersection`, `ground` -> `ground`
     * `house`, `houses`, `building`, `buildings`, `school`, `station`, `mall` -> `building`
   * `street lamp`, `traffic light`, `lamp post` 这类设施**不要**归到 `ground`。

2. **空间关系提取规则 (纯文本摘抄)**：
   * **严禁进行任何视角的逆向推理或逻辑反转**。
   * **空间描述 (spatial_relation)**: 直接原样摘抄原文中出现在该客体附近的方位词、介词短语或连接词（例如："next to", "in front of", "between", "on the intersection of"）。
   * **关系类型 (relation_type)**: 根据摘抄的词汇分类：`directional` (方向), `topological` (拓扑/包含), `ordinal` (序数), `distance` (距离), `composite` (复合)。
   * **参照点 (reference_anchor)**: 提取原文中紧跟在方位词后面的参照物实体名字。如果原文没有明确写出，填 `""`。
   * `it / they / them / this / that` 这类代词**不要**作为有效 `reference_anchor`，直接填 `""`。
   * 如果短语只是属性修饰（如 `building with brown walls`, `house with white roof`），主参照物应提取为 `building`，不要把 `walls` / `roof` 单独当作空间客体。

3. **主体位置 (main_object)**：
   * 如果文中直接写了主体的绝对位置（如 "on Holford Drive"），原样填入 `spatial_relation`。否则留空 `""`。

4. **未知值处理与结构一致性**：
   * 任何无法从文本中明确提取的信息，**必须**填入空字符串 `""`，禁止编造。
   * 输出格式必须是纯 JSON，不包含任何 Markdown 标记（如 ```json）。
   * `main_object` 的 `id` 固定为 `"main"`。`other_objects` 中的 `id` 依次为 `"object1"`, `"object2"`...

# Output JSON Structure
{
  "main_object": {
    "id": "main",
    "category": "ground | building | parking | vehicle | truck | highvegetation | fence | lightpole",
    "spatial_relation": "",
    "relation_type": "",
    "reference_anchor": ""
  },
  "other_objects": [
    {
      "id": "object1",
      "category": "ground | building | parking | vehicle | truck | highvegetation | fence | lightpole",
      "spatial_relation": "直接摘抄原文的方位词短语",
      "relation_type": "directional|topological|ordinal|distance|composite",
      "reference_anchor": "方位词所指的参照物原文"
    }
  ]
}

# Example

**输入：**
"A red car parked between a white car and a blue car in the parking lot of the West Midlands Police Custody Suite."

**输出：**
{
  "main_object": {
    "id": "main",
    "category": "car",
    "spatial_relation": "parked between",
    "relation_type": "directional",
    "reference_anchor": "a white car and a blue car"
  },
  "other_objects": [
    {
      "id": "object1",
      "category": "car",
      "spatial_relation": "between",
      "relation_type": "directional",
      "reference_anchor": "the red car"
    },
    {
      "id": "object2",
      "category": "car",
      "spatial_relation": "between",
      "relation_type": "directional",
      "reference_anchor": "the red car"
    },
    {
      "id": "object3",
      "category": "parking",
      "spatial_relation": "in",
      "relation_type": "topological",
      "reference_anchor": ""
    },
    {
      "id": "object4",
      "category": "building",
      "spatial_relation": "of",
      "relation_type": "topological",
      "reference_anchor": "the parking lot"
    }
  ]
}

# User Input
{description}
"""

# ================= API 配置区域 =================
# 配置: 阿里云百炼
API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = "deepseek-chat"

# ================= 多进程配置 =================
NUM_WORKERS = 8  # 并发进程数，建议4-8，避免API限流
BATCH_SIZE = 10  # 每批处理数量，用于进度显示

def extract_json_content(text: str) -> Optional[Dict[str, Any]]:
    """尝试从文本中提取合法的 JSON 对象。"""
    if not text:
        return None
    
    clean_text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s*```$', '', clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()
    
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

def infer_single(item_data: Dict, api_key: str, base_url: str, model_name: str) -> Dict:
    """
    处理单个数据项的推理
    在子进程中执行，每个进程有自己的 API 客户端
    """
    # 在子进程中导入和创建客户端
    from openai import OpenAI
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    scene_id = str(item_data.get('scene_id', 'unknown'))
    original_object_id = str(item_data.get('object_id', 'unknown'))
    description = item_data.get('description', '')
    
    prompt_text = PROMPT_TEMPLATE.replace("{description}", description)
    messages = [{"role": "user", "content": prompt_text}]
    
    max_retries = 2
    result = None
    
    for attempt in range(max_retries + 1):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.8,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

            raw_text = chat_response.choices[0].message.content.strip()
            parsed_data = extract_json_content(raw_text)
            
            if parsed_data and "main_object" in parsed_data:
                if "other_objects" not in parsed_data:
                    parsed_data["other_objects"] = []
                result = parsed_data
                break
            else:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                    
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
            else:
                print(f"  [错误] Scene {scene_id} 调用失败: {e}")
    
    # 构建 construction 列表
    construction_list = []
    
    # 仅保留空间关系相关字段
    def build_object_props(obj_data: Dict, is_main: bool = False) -> Dict:
        return {
            "category": obj_data.get('category', ''),
            "spatial_relation": obj_data.get('spatial_relation', ''),
            "relation_type": obj_data.get('relation_type', ''),
            "reference_anchor": obj_data.get('reference_anchor', ''),
            "is_main": is_main
        }
    
    if result:
        main_obj = result.get('main_object', {})
        if main_obj:
            main_props = build_object_props(main_obj, is_main=True)
            construction_list.append(main_props)
        
        other_objs = result.get('other_objects', [])
        for idx, obj in enumerate(other_objs):
            if not obj:
                continue
            sub_props = build_object_props(obj, is_main=False)
            sub_props['sub_index'] = idx
            construction_list.append(sub_props)
    
    # 构建输出项
    new_item = item_data.copy()
    new_item['construction'] = construction_list
    
    return new_item

def process_dataset_multiprocess(input_path: str, output_path: str, num_workers: int = 4):
    """使用多进程处理数据集，实时写入结果"""
    
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 {input_path}")
        return

    print(f"正在加载输入文件: {input_path} ...")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    print(f"共加载 {len(all_data)} 条原始数据")
    print(f"使用 {num_workers} 个进程并发处理...")
    print(f"结果将实时保存到: {output_path}")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 准备部分应用函数（传递 API 配置）
    infer_with_config = partial(
        infer_single,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME
    )
    
    completed = 0
    failed = 0
    
    # 打开文件用于实时写入
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 使用进程池
        with Pool(processes=num_workers) as pool:
            # imap 保持顺序，每条处理完立即返回
            for i, result in enumerate(pool.imap(infer_with_config, all_data)):
                completed += 1
                
                if not result.get('construction'):
                    failed += 1
                
                # 实时写入文件
                json.dump(result, f_out, ensure_ascii=False)
                f_out.write('\n')
                f_out.flush()  # 立即刷新到磁盘
                
                # 进度显示
                if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(all_data):
                    progress = (i + 1) / len(all_data) * 100
                    print(f"进度: {i+1}/{len(all_data)} ({progress:.1f}%) | "
                          f"成功: {completed - failed} | 失败/空: {failed}")
    
    print("=" * 60)
    print(f"\n✅ 处理完成！")
    print(f"   总计: {len(all_data)}")
    print(f"   成功: {completed - failed}")
    print(f"   失败: {failed}")
    print(f"   保存至: {output_path}")

if __name__ == "__main__":
    input_files = [
        "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_ND.json",
        "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_new.json",
    ]
    output_dir = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_final"

    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{input_name}_infer_result.jsonl")
        process_dataset_multiprocess(input_file, output_file, num_workers=NUM_WORKERS)
