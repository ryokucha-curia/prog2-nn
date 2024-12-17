import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


#データセットの前処理を定義
ds_transform=transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])

#データセットの読み込み
ds_train=datasets.FashionMNIST(
    root='data',
    train=False,#訓練用を指定
    download=True,
    transform=ds_transform
)

ds_test=datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,#テスト用を指定
    transform=ds_transform
)

#ミニバッチに分割する DataLoaderを作る
batch_size=64
dataloader_train=torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test=torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)

#モデルのインスタンスを作成
model=models.MyModel()

#精度を計算する
acc_test=models.test_accuracy(model,dataloader_test)
print(f'test accurasy:{acc_test*100:.3f}%')

#モデルをインスタンス化する
model=models.MyModel()

#損失関数（誤差関数・ロス関数）の選択
loss_fn=torch.nn.CrossEntropyLoss()

#最適化の方法の選択
learning_rate=1e-3#学習率
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate )

#精度を学習
acc_test=models.test_accuracy(model,dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

#学習回数
n_epochs=20

loss_train_history=[]
loss_test_history=[]
acc_train_history=[]
acc_test_history=[]

#学習
for k in range(n_epochs):
    print(f'epoch{k+1:2}/{n_epochs}',end=': ',flush=True)

    #　１　epochの学習
    time_start=time.time()
    loss_train=models.train(model,dataloader_test,loss_fn,optimizer)
    time_end=time.time()
    loss_train_history.append(loss_train)
    print(f'train loss: {loss_train:.3f} ({time_end-time_start:.1f}s)', end=', ')
    
    time_start=time.time()
    loss_test=models.test(model,dataloader_test,loss_fn)
    time_end=time.time()
    loss_test_history.append(loss_test)
    print(f'test loss: {loss_test:.3f}({time_end-time_start:.1f}s)', end=', ')

    if(k+1)%5==0:
        time_start=time.time()
        acc_train=models.test_accuracy(model,dataloader_train)
        time_end=time.time()
        acc_train_history.append(acc_train)
        print(f'train accuracy: {acc_train*100:.3f}%({time_end-time_start:.1f}s)', end=', ')

        time_start=time.time()
        acc_test=models.test_accuracy(model,dataloader_test)
        time_end=time.time()
        acc_test_history.append(acc_test)
        print(f'test accuracy: {acc_test*100:.3f}%({time_end-time_start:.1f}s)')

#プロット
plt.plot(acc_train_history, label='train')
plt.plot(acc_test_history, label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(loss_train_history, label='train')
plt.plot(loss_test_history, label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.show()