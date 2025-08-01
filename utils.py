
#%%
import torch 
import torch.nn.functional as F
#%%
def create_patchs (x, patch_size, stride):
    """
    x: [batch_size, seq_len, input_dim] # 96 * 7 
    returns: [batch_size, num_patches, patch_size, input_dim]
    """

    B, L, C = x.shape
    num_patches = (L - patch_size) // stride + 1
    patches = [] # input 데이터
    
    for i in range(num_patches): # 시계열을 일정 길이의 패치로 잘라내는 슬라이딩 윈도우 방식 
        start = i * stride # stride = 윈도우 이동 간격 
        end = start + patch_size # patch_size = 한 패치에 포함될 시계열 길이 
        if end > L:
            continue
        patch = x[:, start:end, :] # 모든 배치에 대해서, start부터 end까지 잘래내고, 모든 feature를 사용하자. 
        patches.append(patch.unsqueeze(1)) # 해당 dim의 위치에 크기가 1인 차원을 추가함. 
    return torch.cat(patches, dim = 1) # num_patches 부분이 생김. patch의 개수. 특정 스케일에 해당하는 patch가 몇개인지 
#%%
def hybrid_loss(pred, target):
    """
    pred, target: [batch_size, pred_len, 1]
    returns: scalar hybrid loss (MSE + MAE)
    """
    mse = F.mse_loss(pred,target)
    mae = torch.abs(pred-target).mean()
    return mse + mae, mse, mae

# %%
x = torch.randn(2,96,7)
patch_size = 12
stride = 6
patches = create_patchs(x, patch_size, stride)
print(x)
print("입력 shape:", x.shape)
print("패치 결과 shape:", patches.shape)
print("패치 개수 (num_patches):", patches.shape[1])
# %%
