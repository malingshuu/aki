config = {
    # 数据路径
    'TRAIN_IMG_PATH': 'd:/aaa/bb/bb/data/images',
    'DATA_CSV':'d:/aaa/bb/bb/data/data.xlsx',
    'SELECTED_COLS': ['年龄','高血压','糖尿病','是否独肾',
                      '血糖（术前）', '尿素氮（术前）','乳酸（术前）','肌酐（术前）'
                      , 'AST（术前）','中性粒细胞（术前）','白细胞（术前）'],

    # 模型和结果路径
    'MODEL_SAVE_PATH': 'd:/aaa/bb/bb/model/aki_model.pt',
    'TAB_SAVE_PATH': 'd:/aaa/bb/bb/model/tab_mlp.pkl',
    'IMG_SAVE_PATH': 'd:/aaa/bb/bb/model/img_cnn.pt',
    'MODEL_PATH':'d:/aaa/bb/bb/model/aki_model.pt',
    'RESULT_PATH': 'd:/aaa/bb/bb/result',
    # 训练参数
    'BATCH_SIZE': 8,
    'EPOCHS': 300,
    'LEARNING_RATE': 0.0001,
    'FOCAL_GAMMA': 0.0,
    'WEIGHT_FACTOR': 1.0,
    # 其他参数
    'RANDOM_SEED': 42
}

import os
MODEL_PATH    = 'd:/aaa/bb/bb/model/aki_model.pt'
RESULT_PATH   = 'd:/aaa/bb/bb/result'
DEVICE        = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'


SELECTED_COLS = ['年龄','高血压','糖尿病','是否独肾',
                '血糖（术前）', '尿素氮（术前）','乳酸（术前）','肌酐（术前）'
                ,'AST（术前）','中性粒细胞（术前）','白细胞（术前）']
SHAP_THRESHOLD = 0.1

config['IMG_SIZE'] = 224
config['IMG_MEAN'] = [0.485, 0.456, 0.406]
config['IMG_STD']  = [0.229, 0.224, 0.225]