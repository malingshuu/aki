# shap/1_load_model.py
import sys, os
# 把项目根目录插到最前面，确保能导入 dual_input_model
sys.path.insert(0, os.path.abspath('..'))

import torch
from config import config
from dual_input_model import MultiModalFewShotNet

# 1) 重建模型结构并加载训练好的权重
model = MultiModalFewShotNet(tabular_input_dim=len(config['SELECTED_COLS']))
model.load_state_dict(torch.load(config['MODEL_PATH'], map_location='cpu'))
model.eval()

# 2) 提取并保存表格分支（tab_encoder）
tabular_encoder = model.tab_encoder
torch.save(tabular_encoder.state_dict(), 'tabular_encoder.pt')
print("[OK] tabular_encoder 权重已保存")
