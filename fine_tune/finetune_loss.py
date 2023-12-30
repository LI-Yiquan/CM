"""
Code from https://github.com/sthalles/SimCLR/blob/master/simclr.py
"""
import torch
import torch.nn.functional as F

class FineTuneLoss():
        
        def info_nce_loss(self, features, temperature):

                labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
                labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
                labels = labels.to(self.args.device)

                features = F.normalize(features, dim=1)

                similarity_matrix = torch.matmul(features, features.T)
                # assert similarity_matrix.shape == (
                #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
                # assert similarity_matrix.shape == labels.shape

                # discard the main diagonal from both: labels and similarities matrix
                mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
                labels = labels[~mask].view(labels.shape[0], -1)
                similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
                # assert similarity_matrix.shape == labels.shape

                # select and combine multiple positives
                positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

                # select only the negatives 
                negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

                logits = torch.cat([positives, negatives], dim=1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

                logits = logits / temperature
                return logits, labels

        def pure_contrastive_loss(features, temperature, device):
                
                batch_size = features.shape[0]/2
                n_views = 2
                labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
                labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
                labels = labels.to(device)

                features = F.normalize(features, dim=1)

                similarity_matrix = torch.matmul(features, features.T)

                # discard the main diagonal from both: labels and similarities matrix
                mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
                labels = labels[~mask].view(labels.shape[0], -1)
                similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
                # assert similarity_matrix.shape == labels.shape

                # select and combine multiple positives
                positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

                # select only the negatives 
                negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

                logits = torch.cat([positives, negatives], dim=1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

                logits = logits / temperature
                
                return logits, labels
                
                

        def brier_loss(self, batch, label):
                
                self.p0 = [0.5, 0.6, 0.7, 0.8, 0.9]
                
                logits = self.encoder(batch)
                max_p = torch.max(logits)
                one_hot_label = F.one_hot(label)
                if self.classifier(batch) == label or max_p < self.p0:
                        return (logits - one_hot_label)**2 
                else:
                        return torch.zeros()
        
                
        def ac_loss(self, y, y_hat1, y_hat2, p1, p2):
                
                
                indicator = (y_hat1 == y_hat2) & (y_hat1 != y)

                p1_norm = torch.norm(p1 - p1.detach(), p=2, dim=1) ** 2 
                p2_norm = torch.norm(p2, p=2, dim=1) ** 2

                loss = torch.sum(indicator * (p1_norm + p2_norm))

                return loss

