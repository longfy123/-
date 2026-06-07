import json

# 中英文类别映射（基于上海的英文类别）
CATEGORY_MAPPING = {
    "购物消费": "Shopping & Retail",
    "餐饮美食": "Food & Dining",
    "生活服务": "Daily Services",
    "公司企业": "Offices & Enterprises",
    "交通设施": "Transportation",
    "科教文化": "Education & Culture",
    "商务住宅": "Residential & Business",
    "汽车相关": "Automotive",
    "医疗保健": "Healthcare",
    "酒店住宿": "Hotels",
    "休闲娱乐": "Leisure & Entertainment",
    "金融机构": "Finance",
    "旅游景点": "Tourism",
    "运动健身": "Sports & Fitness"
}

def translate_poi_json():
    """将nanjing_poi.json中的中文类别翻译为英文"""
    input_path = '/root/MoELLM/data/nanjing/nanjing_poi.json'
    output_path = '/root/MoELLM/data/nanjing/nanjing_poi.json'

    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 翻译每个基站的数据
    for base_name, base_info in data.items():
        # 翻译 poi_category_stats 的键
        if 'poi_category_stats' in base_info:
            old_stats = base_info['poi_category_stats']
            new_stats = {}
            for cn_cat, count in old_stats.items():
                en_cat = CATEGORY_MAPPING.get(cn_cat, cn_cat)  # 如果没有映射，保持原样
                new_stats[en_cat] = count
            base_info['poi_category_stats'] = new_stats

        # 翻译 poi_list 中每个POI的category
        if 'poi_list' in base_info:
            for poi in base_info['poi_list']:
                if 'category' in poi:
                    cn_cat = poi['category']
                    poi['category'] = CATEGORY_MAPPING.get(cn_cat, cn_cat)

    # 保存翻译后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ 翻译完成！已更新 {output_path}")
    print(f"✓ 共处理 {len(data)} 个基站")

if __name__ == "__main__":
    translate_poi_json()
