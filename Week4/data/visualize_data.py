import json
import matplotlib.pyplot as plt
from collections import Counter

with open("data/mapping_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

categories = [item["category"] for item in data]
category_counts = Counter(categories)

plt.figure(figsize=(10, 6))
plt.bar(category_counts.keys(), category_counts.values(), color="skyblue")
plt.xlabel("Thể loại")
plt.ylabel("Số lượng bài báo")
plt.title("Số lượng bài báo theo từng thể loại")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("data/category_distribution.png")
