import pandas as pd

# 读取Excel文件
file_path = './1.xlsx'  # 这里修改为你的文件路径
df = pd.read_excel(file_path)

# 需要处理的指标列表（术前和术后）
indicators = [
    '血红蛋白（术前）', '血红蛋白（术后）',
    '白细胞（术前）', '白细胞（术后）',
    'AST（术前）', 'AST（术后）',
    'ALP（术前）', 'ALP（术后）',
    '钠（术前）', '钠（术后）',
    '钾（术前）', '钾（术后）',
    '红细胞比容（术前）', '红细胞比容（术后）',
    '血红蛋白（术前）', '血红蛋白（术后）',
    '血小板（术前）', '血小板（术后）',
    '白细胞（术前）', '白细胞（术后）',
    '中性粒细胞（术前）', '中性粒细胞（术后）',
]

# 创建一个字典来存储每个指标的差值
difference_dict = {}

# 遍历每对指标，计算术前和术后的差值
for i in range(0, len(indicators), 2):
    pre_col = indicators[i]  # 术前列
    post_col = indicators[i+1]  # 术后列
    # 确保列数据为数值类型（转换为float类型），如果无法转换则填充为NaN
    df[post_col] = pd.to_numeric(df[post_col], errors='coerce')
    df[pre_col] = pd.to_numeric(df[pre_col], errors='coerce')

    # 然后再进行差值计算
    difference = df[post_col] - df[pre_col]

    # 计算差值并添加到字典
    difference = df[post_col] - df[pre_col]
    avg_diff = difference.mean()  # 计算平均差值
    
    # 存储指标及其差值平均值
    difference_dict[f'{pre_col} 和 {post_col}'] = avg_diff

# 按照差值平均值降序排序并选出前5个
sorted_diff = sorted(difference_dict.items(), key=lambda x: x[1], reverse=True)

# 输出变化最大的5个指标
print("变化最大的5个指标：")
for idx, (indicator, avg_diff) in enumerate(sorted_diff[:5], start=1):
    print(f"{idx}. {indicator}: 平均差值 = {avg_diff:.2f}")
