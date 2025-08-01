#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import create_patchs # patch 생성 함수 

#%%
# Transformer Backbone : 
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers):
        super().__init__()
        # Multi-Head Attention + Add&Norm 단계 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)  # [B, L, C] 형식의 입력을 받겠다는 설정
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    
    def forward(self, x):  # [B, num_patches, patch_size * C]
        return self.encoder(x)


class TransformerEncoderBranch(nn.Module): # 각 branch를 구성함. 한 branch에서 특정 스케일의 patch를 입력 받아 Transformer를 거쳐 예측 수행하는 구조
    def __init__(self, patch_size,  stride, input_dim, d_model=64, n_heads=4, num_layers=2):
        super().__init__() # 모듈 초기화
        self.patch_size = patch_size
        self.stride = stride
        self.input_dim = input_dim
        self.d_model = d_model # 임베딩 차원
    
        # projection 
        self.project = nn.Linear(patch_size * input_dim, d_model) # patch 하나를 평탄화해서 길이 patch_size *input_dim 벡터로 만들고 d_model로 매핑하는 형식 
        self.encoder = TransformerEncoder(d_model, n_heads, num_layers) # multi-head attention + ffn 반복 patch간의 관계를 학습
        self.output = nn.Linear(d_model, 1) # d_model 차원의 벡터를 최종 예측값으로 변환하는 선형 layer 

    ###  create_patchs -> projection nn.Linear -> embedding self.encoder -> TransformerEncoder (Multi-Head Attention + Add & Norm) ->
    def forward(self, x):  # x: [B, L, C]
        patches = create_patchs(x, self.patch_size, self.stride)  # [B, num_patches, patch_size, C] 여러 branch에서 다른 크기와 stride로 patch 분할하기
        B, N, P, C = patches.shape 
        patches = patches.view(B, N, P * C)  # [B, N, P*C] 
        enc = self.encoder(self.project(patches))  # [B, N, d_model] # 임베딩 하는 과정 
        return self.output(enc)  # [B, N, 1]
    

# 여러 branch 통합 및 최종 예측 수행
class MP3Net(nn.Module):
    def __init__(self, patch_sizes, stride_sizes, pred_len = 24, input_dim=7, d_model=64, n_heads=4, num_layers=2, output_dim=1):
        super().__init__()
        assert len(patch_sizes) == len(stride_sizes), "Patch and stride list lengths must match."
        self.pred_len = pred_len

        # multi sclae parallel branches 생성하기 
        self.branches = nn.ModuleList([
            TransformerEncoderBranch(p, s, input_dim, d_model, n_heads, num_layers) #branch 생성 -> 각 patch branch 병렬 처리 
            for p, s in zip(patch_sizes, stride_sizes)
        ])

        self.fusion_weights = nn.Parameter(torch.ones(len(patch_sizes))) # 모든 branch의 weight를 1로 시작하고, nn.Parameter로 역전파 도중 업데이트됨.
        self.linear = nn.Linear(1, output_dim) # 각 branch에서 예측한 출력값들을 가중합한 결과에 대해 최종 예측값으로 변환하는 선형 layer

    # adaptive fusion module : self.fusion_weights + torch.stack() + softmax() -> 여러 branch 출력을 가중합 
    def forward(self, x):
        outputs = []
        for i, branch in enumerate(self.branches):
            out = branch(x)  # [B, N_i, 1] # branch별 예측값!!
            cur_len = out.shape[1]

            if cur_len < self.pred_len:
                # 마지막 timestep을 복제해서 부족한 길이만큼 padding
                pad = out[:, -1:, :].repeat(1, self.pred_len - cur_len, 1)
                out = torch.cat([out, pad], dim=1)
            elif cur_len > self.pred_len:
                out = out[:, :self.pred_len, :]
                
            outputs.append(out)  # [B, pred_len, 1]

        fused = torch.stack(outputs, dim=0)  # [k, B, pred_len, 1]
        weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1, 1)
        fused = (weights * fused).sum(dim=0)  # [B, pred_len, 1]
        return self.linear(fused)  # [B, pred_len, output_dim] -> 최종 출력 시계열 예측값 = output series 



# %%
