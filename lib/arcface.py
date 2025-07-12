import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EmbeddedBackbone(nn.Module):
    def __init__(self, backbone, num_classes, num_unfreeze = -1, emb_dim = 512):
        super().__init__()
        self.backbone = backbone
        self.embeddings_dim = emb_dim
        self.num_classes = num_classes
        in_features = backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.embeddings = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_features, self.embeddings_dim, bias=True),
            nn.BatchNorm1d(self.embeddings_dim),
        )

        self.classificator = nn.Linear(self.embeddings_dim, self.num_classes)

        if num_unfreeze >= 0 :
            for param in self.backbone.parameters():
                param.requires_grad = False

            if num_unfreeze > 0:
                resnet_blocks = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
                for block in resnet_blocks[-num_unfreeze:]:
                    for param in block.parameters():
                        param.requires_grad = True


    def forward(self, x):
        x = self.backbone(x)
        x = self.embeddings(x)
        x = self.classificator(x)
        return(x)

class ArcFaceLayer(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.3, easy_margin=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # input: (B, in_features)
        # label: (B,)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # (B, C)
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

    def forward_logits(self, x):
        x = F.normalize(x)
        W = F.normalize(self.weight, dim=1)
        return torch.matmul(x, W.T)  # чистый cos(θ), без margin

class FaceNetArc(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        num_classes = backbone.classificator.out_features
        self.backbone = backbone
        self.backbone.classificator = nn.Identity()  # отключаем стандартный классификатор

        self.arc_margin = ArcFaceLayer(512, num_classes)

    def forward(self, x, labels=None):
        x = self.backbone(x)            # (B, 512)

        if labels is not None:
            logits = self.arc_margin(x, labels)  # ArcFace logits
            return logits
        else:
            return F.normalize(x)       # Для инференса

    def forward_logits(self, x):
        return self.arc_margin.forward_logits(x)