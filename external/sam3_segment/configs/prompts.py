# prompts.py

# ==============================================================================
# SENSATURBAN 提示词分发库 (v75 Surgical Repair Edition)
# 
# [优化日志 - 基于全局 IoU 诊断]
# 1. [Fix Ground Blindness]: 增加大量泛化词 ("grey surface", "flat area") 以防止漏检。
# 2. [Fix Water Flooding]: 彻底移除 "black/dark" 等颜色词，防止误吞沥青路和阴影。
# 3. [Fix Rail/Bridge Hallucination]: 移除 "rusty metal" 等泛化材质词，聚焦几何拓扑。
# 4. [Refine]: 微调 Vegetation 防止过度侵蚀地面。
# ==============================================================================

# ================= 基础词库定义 =================

# --- [A] 远景骨架 ---

PROMPTS_ROAD = [
    "asphalt road",           
    "driving street",           
    "traffic lane",             
    "highway",              
    "grey tarmac road",               
    "road pavement",
    "traffic road",
    "street with lane markings",
    "traffic intersection",
    "intersection"
    "long traffic road",
    "long traffic street"
]

PROMPTS_FOOTPATH = [
    "narrow footpath",
    "narrow pedestrian walkway",
    "narrow paved alleyway",
    "narrow brick path",
    "narrow pavement for walking",
    "narrow soil path",
    "narrow concrete path"
]

PROMPTS_GROUND = [
    # --- 核心修复：增加泛化词，解决 100% Blind 问题 ---
    "concrete ground",                 
    "paved plaza",                    
    "public square",               
    "gravel ground",
    "dirt and soil",
    "muddy ground",
    "bare earth",
    "sports court",
    "athletics track",
    "athletics stadium",
    "sports field",
    # [v75 新增：渲染图/低纹理特征兜底]
    "grey surface",             # 针对无纹理水泥
    "flat dark ground",         # 针对深色沥青区
    "textured floor",           # 针对噪点
    "low poly ground",          # 针对渲染风格
    "grey background area",     # 兜底
    "flat open space",
    "urban ground level",
    "lawn",
    "grass"
]

PROMPTS_BUILDING_ROOF = [
    # 保持原有高召回率词条，不做大改
    'building',
    'folded plate roof',            
    'zig-zag roof',                 
    'multi-gabled roof',            
    'series of pointed roofs',      
    'sawtooth roof',                
    'ridged roof',                  
    'greenhouse',                   
    'glass conservatory',           
    'glass roof structure',         
    'metal roof frame',             

    # 材质
    'blue metal roof',              
    'turquoise roof',               
    'light blue roofing',           
    'copper roof',                  
    'green metal roof',             
    'oxidized copper roof',         
    'patchwork roof',               
    'red clay tiles',               
    'terracotta roof',              
    'brown tiled roof',             
    'slate roof',                   
    'grey shingles',                
    'dark grey roof',               
    'corrugated metal roof',        
    'sheet metal roofing',          

    # 结构
    'flat roof with parapet',       
    'parapet wall',                 
    'concrete roof slab',           
    'bitumen roof',                 
    'lift overrun',                 
    'chimney',                      
    'dormer window',                
    'skylight',                     
    'mansard roof',                 

    # 类型
    'warehouse',                    
    'factory building',             
    'residential house',            
    'terraced housing',             
    'commercial building block',    
    'office block',                 

    # 立面防误触
    'building facade',              
    'side of a building',           
    'brick wall of a house',        
    'concrete building wall',       
    'apartment block facade',       
    'multi-story building',

    'whole building structure',             # 强调“结构完整性”
    'freestanding architectural structure', # 强调“独立性”
    'complete house from above',            # 强调“视角”
    
    # 2. 强调轮廓与体积 (Volume & Footprint)
    'rectangular building block',           # 强调几何块状感
    'large scale industrial complex',       # 针对大厂房的整体描述
    'urban housing block',                  # 针对密集住宅块
    
    # 3. 强调“建筑群”或“连接性” (防止把连体建筑切开)
    'connected building',
    'row of attached houses',               # 针对联排/握手楼
]

PROMPTS_VEG_CANOPY = [
    "tree canopy", "crown of a tree", "top view of trees",
    "forest", "woods", "grove", "cluster of trees",
    "dense foliage", "leafy tree top",
    "sparse tree canopy", "leafless trees", 
    "tree branches structure", 
    "bushes", "hedges", "shrubbery", "row of hedges",
    "large leafy bush",
    "cluster of green trees",
]

PROMPTS_WATER = [
    # [v75 核心修复]: 彻底移除 "black/dark" 词汇，解决 Water 召回率高但精度极低的问题
    # 只要不提及“黑色”，就不会把沥青路和阴影吃掉
    
    # 1. 物理水体
    "river water", 
    "canal channel",
    "flowing river stream",
    "wide river basin",
    "lake surface",
    
    # 2. 脏水/自然特征 (保留但不强调颜色)
    "muddy river bank",
    "brown sludge on water",
    "green duckweed",               
    "algae floating on water",      
    "swampy water",
    "water with ripples",
    
    # 3. 边界 (小心使用)
    "wet mud along river edge",
    "mossy water edge"
    
    # [已删除的高危词]:
    # "black winding path" (这是路!)
    # "dark void" (这是阴影!)
    # "reflective dark surface" (这是车顶或路!)
    # "black river water" (风险太大)
]

# --- [D] 结构体 (高误报区) ---

PROMPTS_BRIDGE = [
    "bridge",
    # "oversection of highway",
    "overriver section of highway"
    # 移除了单字 "bridge"，太容易误触
]

PROMPTS_PARKING = [
    "parking lot",                                      
    "wide space to parking",
    "wide surface to parking",              
    "asphalt parking area",                                 
    "parking area",
    "parking surface",
    "parking spaces"
]

# --- [F] 近景物体 (大幅精简) ---

PROMPTS_RAIL = [
    # [v75 修复]: 移除材质词，聚焦几何。解决 93% 误报率。
    
    # 1. 几何强特征
    "railway tracks",
    "train lines on the ground",
    "pair of parallel steel rails", # 强调成对
    "railroad ties and sleepers",     # 强调枕木
    # [已删除的高危词]:
    "rusty metal tracks", 
    "two parallel steel lines"
]

PROMPTS_FURNITURE = [
    "dumpster", "waste skip", "industrial trash bin", "wheelie bin",
    "shipping container", "cargo container", 
    "long ship",
    "boat",
    "portable toilet", "porta potty", 
    "utility box", "electrical cabinet", 
    "grit bin", "roadside bin", 
    "pile of debris", "rubble pile", "construction waste",
    "stack of pipes", "piled PVC pipes",
    "wooden pallet", "stack of pallets",
    "HVAC unit", 
    "ventilation grille", "solar panel array",
    "scaffolding", "safety netting",
    "satellite dish", "antenna", 
    "patio furniture", "picnic table", "parasol", 
    "playground equipment", "slide", "swing", "trampoline", 
    "garden shed", "greenhouse", 
    "narrowboat", "barge", "moored vessel",  "sewer cover",  "bollard", 
    "public bench", "bus stop shelter", "billboard",
    "construction material stack",
    "construction materials scattered around",
    "construction debris",
    "pallets and industrial waste",

    # 2. 堆积物 (针对中间沙堆 - 重点区分 Ground)
    "cone-shaped pile of sand",       # 强调圆锥几何
    "mound of grey aggregate",        # 强调“小丘”形态
    "heap of earth",                  # 强调“堆”
    
    # 3. 线性堆叠 (针对左下管道 & 左上木材)
    "stack of black cylindrical pipes", # 强调“圆柱”+“黑色”
    "pile of wooden planks",            # 强调“板材”
    "lumber stacked neatly",            # 强调“堆叠”
    
    # 4. 高反差点状物 (针对白色袋子)
    "white industrial sacks",
    "cluster of white sandbags",
]

PROMPTS_WALL = [
    # [v75 优化]: 强调线性特征，试图与 Building 区分
    "brick wall barrier", 
    "stone fence", 
    "wooden fence panel",
    "metal railing fence", 
    "garden wall separation",
    "freestanding wall"
]

PROMPTS_CAR = [
    "car", 
    "van", 
    "truck", 
    "bus",              
    "car roof",
    "wheels",
    "parked car"
]

PROMPTS_BIKE = [
    "bicycle", 
    "motorcycle", 
    "scooter",
    "parked bike"
]

# ================= 视角分发配置 =================

STRICT_PROMPTS = {
    # 1. 远景 (Global)
    "BEV_GLOBAL": {
        "Traffic Road": PROMPTS_ROAD, 
        "Footpath": PROMPTS_FOOTPATH, 
        "Ground": PROMPTS_GROUND,     
        "Building": PROMPTS_BUILDING_ROOF,
        "Vegetation": PROMPTS_VEG_CANOPY,
        # Water 即使在远景移除也不好，容易被地吃掉，保留但词库已净化
        "Water": PROMPTS_WATER,       
        "Bridge": PROMPTS_BRIDGE
    },

    # 2. 近景 (Closeup)
    "BEV_CLOSEUP": {
        "Parking": PROMPTS_PARKING,
        "Wall": PROMPTS_WALL,
        "Rail": PROMPTS_RAIL,
        "Street Furniture": PROMPTS_FURNITURE,
        "Car": PROMPTS_CAR,
        "Bike": PROMPTS_BIKE
    },

    # 3. 全模式 (Tiles)
    "TILE_ALL": {
        "Traffic Road": PROMPTS_ROAD, 
        "Footpath": PROMPTS_FOOTPATH, 
        "Ground": PROMPTS_GROUND,
        "Building": PROMPTS_BUILDING_ROOF,
        "Vegetation": PROMPTS_VEG_CANOPY,
        "Water": PROMPTS_WATER,
        "Bridge": PROMPTS_BRIDGE,
        "Parking": PROMPTS_PARKING,
        "Wall": PROMPTS_WALL,
        "Rail": PROMPTS_RAIL,
        "Street Furniture": PROMPTS_FURNITURE,
        "Car": PROMPTS_CAR,
        "Bike": PROMPTS_BIKE
    }
}

def get_prompts_strict(mode):
    if mode == 'global':
        source = STRICT_PROMPTS["BEV_GLOBAL"]
    elif mode == 'closeup':
        source = STRICT_PROMPTS["BEV_CLOSEUP"]
    elif mode == 'all':
        source = STRICT_PROMPTS["TILE_ALL"]
    else:
        return [], {}

    flat_list = []
    mapping = {}
    for cls_name, txts in source.items():
        for t in txts:
            flat_list.append(t)
            mapping[t] = cls_name
    return flat_list, mapping