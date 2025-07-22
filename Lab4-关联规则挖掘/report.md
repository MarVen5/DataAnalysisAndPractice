# **数据分析及实践 Lab 4**

### **PART 1**

读取数据集 `data.csv`，进行数据预处理。

##### **Q1**
选取问卷中的 `SC155Q01HA`, `SC155Q02HA`, `SC155Q03HA`, `SC155Q04HA`, `SC155Q05HA` 5 个离散性特征作为特征集，分别介绍这些特征所代表的含义和各自取值范围，注意到这些特征名称本身较为冗长且不易于理解，请对特征名进行简化修改，并删除存在缺失值的行。

我们根据 `codebook.xlsx` 内容第 `2216` 到 `2256` 行的说明完成以下任务：

```python
import pandas as pd

# 读取数据集
df = pd.read_csv('data.csv', index_col=0)

# 选取指定特征
df = df[[
    "SC155Q01HA", 
    "SC155Q02HA", 
    "SC155Q03HA", 
    "SC155Q04HA", 
    "SC155Q05HA"
]].copy()

# 定义并应用新列名
df = df.rename(columns={
    "SC155Q01HA": "Network_Devices_Sufficiency",      # 联网数字设备数量充足性
    "SC155Q02HA": "Internet_Bandwidth_Sufficiency",   # 网络带宽 / 速度充足性
    "SC155Q03HA": "Instruction_Devices_Sufficiency",  # 教学用数字设备数量充足性
    "SC155Q04HA": "Device_Performance_Sufficiency",   # 设备计算能力充足性
    "SC155Q05HA": "Software_Availability_Sufficiency" # 软件资源充足性
})

df.dropna(inplace=True)
print("1)\n",df)

# 删除存在缺失值的行
df_clean = df.dropna()

# 查看处理结果
print(f"\n2)\nOriginal data volume: {len(df)} lines")
print(f"Processed data volume: {len(df_clean)} lines")
```
运行结果如下：
```
1)
        Network_Devices_Sufficiency  Internet_Bandwidth_Sufficiency  \
1                              2.0                             2.0   
2                              2.0                             2.0   
3                              2.0                             3.0   
4                              2.0                             3.0   
5                              2.0                             4.0   
...                            ...                             ...   
21899                          2.0                             3.0   
21900                          3.0                             3.0   
21901                          3.0                             3.0   
21902                          4.0                             3.0   
21903                          3.0                             3.0   

       Instruction_Devices_Sufficiency  Device_Performance_Sufficiency  \
1                                  2.0                             1.0   
2                                  2.0                             2.0   
3                                  1.0                             1.0   
4                                  2.0                             2.0   
5                                  2.0                             2.0   
...                                ...                             ...   
21899                              2.0                             2.0   
21900                              2.0                             2.0   
21901                              3.0                             2.0   
21902                              3.0                             3.0   
21903                              2.0                             2.0   

       Software_Availability_Sufficiency  
1                                    1.0  
2                                    2.0  
3                                    1.0  
4                                    2.0  
5                                    2.0  
...                                  ...  
21899                                2.0  
21900                                2.0  
21901                                2.0  
21902                                3.0  
21903                                2.0  

[20681 rows x 5 columns]

2)
Original data volume: 20681 lines
Processed data volume: 20681 lines
```

不同特征的原名、简化名和详细的解释如下（1 表示最差，4 表示最好）：
| 原特征名       | 简化特征名            | 详细解释                                     | 取值范围                                |
| -------------- | --------------------- | -------------------------------------------- | --------------------------------------------- |
| SC155Q01HA     | Network_Devices       | 学校可联网数字设备数量是否满足教学需求       | 1-4              |
| SC155Q02HA     | Internet_Bandwidth    | 学校网络带宽是否支持教学活动                 | 1-4              |
| SC155Q03HA     | Teaching_Devices      | 专用教学设备数量是否充足                     | 1-4              |
| SC155Q04HA     | Device_Performance    | 设备计算性能是否满足教学软件要求             | 1-4              |
| SC155Q05HA     | Software_Resources    | 教学相关软件资源是否充足                     | 1-4              |

##### **Q2**

注意到选取的特征可能存在相同取值（如特征 A 和 B 都可能取值 0），不便于后续的关联分析过程。请构建项集索引，并依据索引内容进行特征值替换。项集索引字典形式如下：
```python
ind2val = { 0: '[COLUMN1]=[VALUE1]', 1: '[COLUMN1]=[VALUE2]', ... , }
```
基于所选项集索引字典进行单元格内容替换，以便于后续频繁项集挖掘和关联分析过程。

```python
ind2val = {
    ('Network_Devices_Sufficiency', 1): 0,
    ('Network_Devices_Sufficiency', 2): 1,
    ('Network_Devices_Sufficiency', 3): 2,
    ('Network_Devices_Sufficiency', 4): 3,
    
    ('Internet_Bandwidth_Sufficiency', 1): 4,
    ('Internet_Bandwidth_Sufficiency', 2): 5,
    ('Internet_Bandwidth_Sufficiency', 3): 6,
    ('Internet_Bandwidth_Sufficiency', 4): 7,

    ('Instruction_Devices_Sufficiency', 1): 8,
    ('Instruction_Devices_Sufficiency', 2): 9,
    ('Instruction_Devices_Sufficiency', 3): 10,
    ('Instruction_Devices_Sufficiency', 4): 11,

    ('Device_Performance_Sufficiency', 1): 12,
    ('Device_Performance_Sufficiency', 2): 13,
    ('Device_Performance_Sufficiency', 3): 14,
    ('Device_Performance_Sufficiency', 4): 15,

    ('Software_Availability_Sufficiency', 1): 16,
    ('Software_Availability_Sufficiency', 2): 17,
    ('Software_Availability_Sufficiency', 3): 18,
    ('Software_Availability_Sufficiency', 4): 19
}

# 替换特征值
for (col, val), idx in ind2val.items():
    df[col] = df[col].replace(float(val), int(idx))

# 输出替换后的结果
print(df)
```
运行结果如下：
```
Network_Devices_Sufficiency  Internet_Bandwidth_Sufficiency  \
1                              1.0                             5.0   
2                              1.0                             5.0   
3                              1.0                             6.0   
4                              1.0                             6.0   
5                              1.0                             7.0   
...                            ...                             ...   
21899                          1.0                             6.0   
21900                          2.0                             6.0   
21901                          2.0                             6.0   
21902                          3.0                             6.0   
21903                          2.0                             6.0   

       Instruction_Devices_Sufficiency  Device_Performance_Sufficiency  \
1                                  9.0                            12.0   
2                                  9.0                            13.0   
3                                  8.0                            12.0   
4                                  9.0                            13.0   
5                                  9.0                            13.0   
...                                ...                             ...   
21899                              9.0                            13.0   
21900                              9.0                            13.0   
21901                             10.0                            13.0   
21902                             10.0                            14.0   
21903                              9.0                            13.0   

       Software_Availability_Sufficiency  
1                                   16.0  
2                                   17.0  
3                                   16.0  
4                                   17.0  
5                                   17.0  
...                                  ...  
21899                               17.0  
21900                               17.0  
21901                               17.0  
21902                               18.0  
21903                               17.0  

[20681 rows x 5 columns]
```

### **PART 2**

基于预处理后的数据集，编写算法代码进行频繁项集挖掘。

##### **Q1**
请参考 Apriori 产生频繁项集的算法流程，自行编写相应代码，分别以最小支持度阈值为 0.25 和 0.5，挖掘频繁项集。

```python
# 将 DataFrame 转为交易列表
data = df[
    ['Network_Devices_Sufficiency',
     'Internet_Bandwidth_Sufficiency',
     'Instruction_Devices_Sufficiency',
     'Device_Performance_Sufficiency',
     'Software_Availability_Sufficiency']
].values.tolist()

# 去掉 NaN
data = [[val for val in row if pd.notna(val)] for row in data]

for row in data:
    print(row)
```
运行结果如下：
```
[1.0, 5.0, 9.0, 12.0, 16.0]
[1.0, 5.0, 9.0, 13.0, 17.0]
[1.0, 6.0, 8.0, 12.0, 16.0]
[1.0, 6.0, 9.0, 13.0, 17.0]
[1.0, 7.0, 9.0, 13.0, 17.0]
[1.0, 5.0, 9.0, 14.0, 17.0]
[2.0, 7.0, 10.0, 14.0, 18.0]
[3.0, 7.0, 11.0, 15.0, 19.0]
[2.0, 7.0, 10.0, 14.0, 18.0]
[2.0, 7.0, 10.0, 15.0, 18.0]
[1.0, 7.0, 8.0, 12.0, 17.0]
[0.0, 6.0, 8.0, 12.0, 17.0]
[1.0, 5.0, 9.0, 13.0, 17.0]
[0.0, 6.0, 8.0, 12.0, 17.0]
[1.0, 5.0, 9.0, 13.0, 17.0]
[3.0, 7.0, 11.0, 15.0, 19.0]
[0.0, 5.0, 8.0, 12.0, 18.0]
[2.0, 6.0, 8.0, 12.0, 17.0]
[3.0, 7.0, 11.0, 14.0, 19.0]
[2.0, 5.0, 11.0, 14.0, 18.0]
[2.0, 6.0, 9.0, 13.0, 17.0]
[1.0, 5.0, 9.0, 13.0, 17.0]
[3.0, 7.0, 11.0, 15.0, 19.0]
[2.0, 7.0, 10.0, 14.0, 18.0]
[0.0, 5.0, 10.0, 14.0, 18.0]
...
[2.0, 6.0, 9.0, 13.0, 17.0]
[2.0, 6.0, 10.0, 13.0, 17.0]
[3.0, 6.0, 10.0, 14.0, 18.0]
[2.0, 6.0, 9.0, 13.0, 17.0]
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```
我们根据实验文档中给出的思路，编写了算法代码如下，并且分别以最小支持度阈值为 0.25 和 0.5 时挖掘频繁项集；
```python
def generate_initial_candidates(records):
    # 生成所有单项集候选项 C1
    unique_items = set()
    for record in records:
        for elem in record:
            unique_items.add(frozenset([elem]))
    return sorted(list(unique_items))


def all_k_minus_1_subsets(item_group, length):
    # 返回所有 (k-1) 子集，用于剪枝
    items = list(item_group)
    return [set(items[:i] + items[i+1:]) for i in range(length)]


def filter_frequent_items(transactions, candidate_sets, threshold):
    # 计算支持度并筛选频繁项集
    item_counter = {}
    total = len(transactions)

    for entry in transactions:
        for cand in candidate_sets:
            if cand.issubset(entry):
                key = tuple(sorted(cand))
                item_counter[key] = item_counter.get(key, 0) + 1

    frequent_items = []
    for key, count in item_counter.items():
        if count / total >= threshold:
            frequent_items.append(set(key))

    return frequent_items


def count_supports(transactions, itemsets):
    # 支持度计数，返回字典 {项集: 支持度}
    support_map = {}
    total = len(transactions)

    for entry in transactions:
        for item in itemsets:
            if item.issubset(entry):
                key = tuple(sorted(item))
                support_map[key] = support_map.get(key, 0) + 1

    return {key: val / total for key, val in support_map.items()}


def join_step(prev_frequent, level):
    # 连接操作，生成候选 k 项集
    result = []
    count = len(prev_frequent)

    for i in range(count):
        for j in range(i + 1, count):
            a = sorted(list(prev_frequent[i]))[:level - 2]
            b = sorted(list(prev_frequent[j]))[:level - 2]
            if a == b:
                merged = prev_frequent[i] | prev_frequent[j]
                result.append(merged)

    return result


def prune_step(candidates, prev_freq_sets, k):
    # 剪枝：检查所有 (k-1) 子集是否都在频繁集中
    filtered = []
    for group in candidates:
        subsets = all_k_minus_1_subsets(group, k)
        if all(sub in prev_freq_sets for sub in subsets):
            filtered.append(group)
    return filtered


def apriori_algorithm(raw_data, min_sup=0.5):
    # Apriori 主流程函数
    dataset = [set(row) for row in raw_data]
    level = 1
    freq_sets_all = []
    support_data_all = []

    # 初始单项集
    base_C1 = generate_initial_candidates(dataset)
    base_F1 = filter_frequent_items(dataset, base_C1, min_sup)
    freq_sets_all.append(base_F1)
    support_data_all.append(count_supports(dataset, base_F1))

    # 从 k=2 开始迭代生成
    while freq_sets_all[level - 1]:
        level += 1
        Ck = join_step(freq_sets_all[level - 2], level)
        Ck = prune_step(Ck, freq_sets_all[level - 2], level)
        Fk = filter_frequent_items(dataset, Ck, min_sup)
        if not Fk:
            break
        freq_sets_all.append(Fk)
        support_data_all.append(count_supports(dataset, Fk))

    return freq_sets_all, support_data_all
```
分别执行当 0.25 和 0.5 时候的结果：
```python
print('1) When the minimum support threshold is 0.25:')
apriori_algorithm(data, min_sup=0.25)
```
```
1) When the minimum support threshold is 0.25:
([[{1.0}, {9.0}, {13.0}, {17.0}, {6.0}, {7.0}, {14.0}, {10.0}, {18.0}, {2.0}],
  [{10.0, 14.0},
   {14.0, 18.0},
   {2.0, 14.0},
   {10.0, 18.0},
   {2.0, 10.0},
   {2.0, 18.0},
   {2.0, 6.0},
   {6.0, 14.0},
   {6.0, 18.0}]],
 [{(1.0,): 0.26816885063584933,
   (9.0,): 0.3363473719839466,
   (13.0,): 0.29447318795029254,
   (17.0,): 0.27334268168850634,
   (6.0,): 0.4188385474590204,
   (7.0,): 0.3350901793917122,
   (14.0,): 0.4361491223828635,
   (10.0,): 0.3700014506068372,
   (18.0,): 0.48063439872346597,
   (2.0,): 0.3985784052995503},
  {(10.0, 14.0): 0.27450316715826123,
   (14.0, 18.0): 0.33054494463517237,
   (2.0, 14.0): 0.26183453411343743,
   (10.0, 18.0): 0.2713118321164354,
   (2.0, 10.0): 0.2744064600357816,
   (2.0, 18.0): 0.27401963154586334,
   (2.0, 6.0): 0.2602872201537643,
   (6.0, 14.0): 0.2694260432280837,
   (6.0, 18.0): 0.2733910352497461}])
```
```python
print('2) When the minimum support threshold is 0.5:')
apriori_algorithm(data, min_sup=0.5)
```
```
2) When the minimum support threshold is 0.5:
([[]], [{}])
```

##### **Q2**
当最小支持度为 0.5 时，频繁项集数量较少。请将各特征原始取值为 1 和 2 的单元格统一修改其值为 0，取值为 3 和 4 的单元格统一修改其值为 1。重复 T1-Q2 的项集索引构建过程，并以最小支持度阈值为 0.5，挖掘频繁项集。
```python
df_transformed = pd.read_csv('data.csv', index_col=0)

# 选取指定特征
df_transformed = df_transformed[[
    "SC155Q01HA", 
    "SC155Q02HA", 
    "SC155Q03HA", 
    "SC155Q04HA", 
    "SC155Q05HA"
]].copy()

# 定义并应用新列名
df_transformed = df_transformed.rename(columns={
    "SC155Q01HA": "Network_Devices_Sufficiency",      # 联网数字设备数量充足性
    "SC155Q02HA": "Internet_Bandwidth_Sufficiency",   # 网络带宽 / 速度充足性
    "SC155Q03HA": "Instruction_Devices_Sufficiency",  # 教学用数字设备数量充足性
    "SC155Q04HA": "Device_Performance_Sufficiency",   # 设备计算能力充足性
    "SC155Q05HA": "Software_Availability_Sufficiency" # 软件资源充足性
})

df_transformed.dropna(inplace=True)

# 1 和 2 替换为 0，3 和 4 替换为 1
df_transformed = df_transformed.replace({1: 0, 2: 0, 3: 1, 4: 1})

ind2val_transformed = {
    ('Network_Devices_Sufficiency', 0): 0,
    ('Network_Devices_Sufficiency', 1): 1,
    
    ('Internet_Bandwidth_Sufficiency', 0): 2,
    ('Internet_Bandwidth_Sufficiency', 1): 3,

    ('Instruction_Devices_Sufficiency', 0): 4,
    ('Instruction_Devices_Sufficiency', 1): 5,

    ('Device_Performance_Sufficiency', 0): 6,
    ('Device_Performance_Sufficiency', 1): 7,

    ('Software_Availability_Sufficiency', 0): 8,
    ('Software_Availability_Sufficiency', 1): 9
}

for (col, val), idx in ind2val_transformed.items():
    df_transformed[col] = df_transformed[col].replace(float(val), int(idx))

# 将每一行转化为列表，并去除 NaN
transactions = df_transformed.values.tolist()
transactions = [[item for item in row if pd.notna(item)] for row in transactions]

print('When the minimum support threshold is 0.5:')
apriori_algorithm(transactions, min_sup=0.5)
```
运行结果如下：
```
When the minimum support threshold is 0.5:
([[{3.0}, {7.0}, {9.0}, {1.0}, {5.0}],
  [{1.0, 3.0}, {7.0, 9.0}, {1.0, 7.0}, {1.0, 9.0}, {1.0, 5.0}]],
 [{(3.0,): 0.6224553938397563,
   (7.0,): 0.5967312992601905,
   (9.0,): 0.6324645810163918,
   (1.0,): 0.625211546830424,
   (5.0,): 0.547507373918089},
  {(1.0, 3.0): 0.5143368309075963,
   (7.0, 9.0): 0.5176248730719017,
   (1.0, 7.0): 0.5007978337604565,
   (1.0, 9.0): 0.5123543348967652,
   (1.0, 5.0): 0.5080025143851845}])
```
##### **Q3**

分析 Q1 和 Q2 的结果，你有什么发现？请根据各特征定义，分析产生这种情况的原因。

ANS

从分析结果可以看出，在接受调查的学校中，有超过一半在与电子设备相关的各项指标上表现良好，说明整体教育信息化水平处于较高水平。通常情况下，如果一个学校拥有较为充足的电子设备资源，这往往意味着其所在地区具备较好的经济基础和教育投入能力。在这样的前提下，学校不仅能配备基本的教学硬件，还更有能力建设稳定高速的网络环境、引进性能更强的计算设备，以及配置更加丰富和实用的教学软件资源。

这一现象表明，学校在信息化建设上的各个方面具有明显的协同发展特征：设备条件的改善常常伴随着网络环境的优化和软件资源的完善，而不是单一方面的提升。背后的原因可能包括政府统一规划的教育现代化政策、地区经济发展水平差异带来的资源可及性，以及学校在信息化推进过程中的整体布局与投资策略。

### **PART 3**

基于 T2-Q2 得到的频繁项集挖掘结果，编写算法代码进行关联规则提取。

##### **Q1**
以最小置信度阈值为 0.8 ，提取形如 `X->{1}` 的关联规则，并输出它们的置信度和提升度。
```
def extract_rules(F_list, support_list, min_conf=0.8):
    rules = []
    for k in range(1, len(F_list)):  # 从 2 项集开始
        Fk = F_list[k]
        support_k = support_list[k]
        support_prev = support_list[k-1]

        for itemset in Fk:
            if 1 not in itemset or len(itemset) < 2:
                continue  # 只考虑含有 {1} 且长度 >=2 的项集

            for item in itemset:
                if item == 1:
                    continue
                antecedent = frozenset([item])
                consequent = frozenset([1])
                combined = tuple(sorted(itemset))
                antecedent_key = tuple(sorted(antecedent))
                support_XY = support_k.get(combined, 0)
                support_X = support_prev.get(antecedent_key, 0)
                support_Y = support_prev.get((1,), 0)

                if support_X == 0 or support_Y == 0:
                    continue

                confidence = support_XY / support_X
                lift = confidence / support_Y

                if confidence >= min_conf:
                    rules.append({
                        "rule": f"{set(antecedent)} → {set(consequent)}",
                        "confidence": round(confidence, 10),
                        "lift": round(lift, 10)
                    })

    return rules

F_list, support_list = apriori_algorithm(transactions, min_sup=0.5)
rules = extract_rules(F_list, support_list, min_conf=0.8)

for r in rules:
    print(f"Rule: {r['rule']}, Confidence: {r['confidence']}, Lift: {r['lift']}")
```
运行结果如下：
```
Rule: {3.0} → {1}, Confidence: 0.826303115, Lift: 1.3216376429
Rule: {7.0} → {1}, Confidence: 0.8392350701, Lift: 1.3423217699
Rule: {9.0} → {1}, Confidence: 0.8100917431, Lift: 1.2957082242
Rule: {5.0} → {1}, Confidence: 0.9278459772, Lift: 1.4840512494
```

##### **Q2**
参考项集索引的对应关系，对以上频繁项集和关联规则结果进行简要分析和总结。

ANS

在最小支持度为 0.5 的条件下所挖掘出的频繁项集表明，在被调查的学校中，多数学校在网络设施建设、教学电子设备配备、计算能力和软件充足性等方面的条件较好。这说明学校在信息化建设方面已经有了相当的基础，具备较为完善的数字教学支持环境。特别是包含值为 1 的特征项频繁出现在不同项集中，意味着超过一半的学校在这些指标上的表现都是较为理想的，反映了当前教育资源在多数地区的基础建设水平较高。

进一步分析关联规则的结果可以发现，当一个学校在某一方面具备良好的设施（例如网络设备充足），那么它在其他相关方面（如软件资源或计算能力）也往往处于较好水平。这些规则通常具有较高的置信度（例如超过 0.8），且提升度大于 1，表明这种共现关系不是随机的，而是存在较强的依赖性。这说明学校在推进信息化建设的过程中，各项基础条件是协同发展的，具备良好经济与政策支持的学校往往能够在多个维度同步提升教育信息化能力。

总结：学校在某一项指标表现良好的情况下，很可能其整体的信息化条件也较为完善。这一发现对于教育资源的评估和规划具有实际意义，可以作为制定教育政策和资源分配方案的参考依据。同时也表明，对于存在短板的学校，可以通过提升其某一个关键指标，带动整体水平的改善，从而进一步推动教育公平和教育信息化的全面发展。
