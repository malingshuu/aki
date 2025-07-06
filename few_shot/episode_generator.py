
import random
import pandas as pd

def create_episode(dataset, n_way=2, k_shot=5, q_query=5):
    selected_classes = random.sample(list(dataset['label'].unique()), n_way)
    support, query = [], []
    label_map = {c: i for i, c in enumerate(selected_classes)}

    for c in selected_classes:
        samples = dataset[dataset['label'] == c]
        chosen = samples.sample(k_shot + q_query)
        support.append(chosen.iloc[:k_shot].assign(label=label_map[c]))
        query.append(chosen.iloc[k_shot:].assign(label=label_map[c]))

    support_set = pd.concat(support).reset_index(drop=True)
    query_set = pd.concat(query).reset_index(drop=True)
    return support_set, query_set
