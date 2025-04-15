import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, cat_index_dict, embedding_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.embedding_list = nn.ModuleList([nn.Embedding(len(cat_index_dict[cat_col]), embedding_dim) for cat_col in cat_index_dict])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, cat_feature_list, non_cat_features):
        bsz=non_cat_features.shape[0]
        embedded_data = torch.cat([self.embedding_list[i](cat_features.to(torch.long)) for i,cat_features in enumerate(cat_feature_list)], dim=1)
        x = torch.cat([embedded_data.view(bsz,-1), non_cat_features], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x