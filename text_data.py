import numpy as np
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, question_data, answer_data, title_data, category_data, 
                 host_data, use_embeddings, dist_features, idxs, targets=None):
        self.question_data = question_data[idxs]
        self.answer_data = answer_data[idxs]
        self.title_data = title_data[idxs]
        self.category_data = category_data[idxs]
        self.host_data = host_data[idxs]
        self.use_embeddings_q = use_embeddings['question_body_embedding'][idxs]
        self.use_embeddings_a = use_embeddings['answer_embedding'][idxs]
        self.use_embeddings_t = use_embeddings['question_title_embedding'][idxs]
        self.dist_features = dist_features[idxs]
        if targets is not None: self.targets = targets[idxs]  
        else: self.targets = np.zeros((self.question_data.shape[0], 30))

    def __getitem__(self, idx):
        question = self.question_data[idx]
        answer = self.answer_data[idx]
        title = self.title_data[idx]
        category = self.category_data[idx]
        host = self.host_data[idx]
        use_emb_q = self.use_embeddings_q[idx]
        use_emb_a = self.use_embeddings_a[idx]
        use_emb_t = self.use_embeddings_t[idx]
        dist_feature = self.dist_features[idx]
        target = self.targets[idx]

        return (question, answer, title, category, host, use_emb_q, use_emb_a, 
                use_emb_t, dist_feature, target)

    def __len__(self):
        return len(self.question_data)