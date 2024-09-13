import json

# 加载数据
with open('top_k_bruteforce.json', 'r') as f:
    bf_topk = json.load(f)
    
with open('top_k_Optimization.json', 'r') as f:
    opt_topk = json.load(f)
    
# 比较数据
count = 0
id = 0
for key in opt_topk:
    if key in bf_topk:
        count += 1
    else:
        print(f"No.{id} - {key} not in bf_topk")
        
    id += 1

print(f"{count} / {len(opt_topk)} = {count / len(opt_topk)}")

count = 0
id = 0
for key in  bf_topk:
    if key in opt_topk:
        count += 1
    else:
        print(f"No.{id} - {key} not in opt_topk")
        
    id += 1

print(f"{count} / {len(bf_topk)} = {count / len(bf_topk)}")

# 找出有多少topk是有2597的 在bf_topk中
count = 0
for key in bf_topk:
    if 2597 in bf_topk[key][0]:
        count += 1
    
print(f"{count} / {len(bf_topk)} = {count / len(bf_topk)}")