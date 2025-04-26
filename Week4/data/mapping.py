import os
import json
import re

# Đường dẫn thư mục chứa các bài báo
NEWS_DIR = os.path.join('data', 'news')

mapping = []

# Regex để lấy id từ tên file (dãy số sau dấu - trước .txt)
id_pattern = re.compile(r'-(\d+)\.txt$')

for category in os.listdir(NEWS_DIR):
    category_path = os.path.join(NEWS_DIR, category)
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            if filename.endswith('.txt'):
                match = id_pattern.search(filename)
                if match:
                    file_id = match.group(1)
                    rel_path = os.path.join('data', 'news', category, filename)
                    mapping.append({
                        'file_path': rel_path.replace('\\', '/'),
                        'category': category,
                        'id': file_id
                    })

# Lưu ra file json
with open('data/mapping_news.json', 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2) 