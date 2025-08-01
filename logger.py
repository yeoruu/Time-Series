#%%
import csv
#%%
def init_metric_log(filepath): # csv 파일을 열고 초기화하고 열 제목 쓰기. 학습 시작 전에 한번 실행
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_mse', 'train_mae',
                         'val_loss', 'val_mse', 'val_mae'])
#%%
def append_metric_log(filepath, epoch, train_loss, train_mse, train_mae,
                      val_loss, val_mse, val_mae):
    with open(filepath, 'a', newline='') as f: # 매 epoch마다 결과를 한 줄씩 metrics.csv에 추가함. 
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_mse, train_mae,
                         val_loss, val_mse, val_mae])
# %%
