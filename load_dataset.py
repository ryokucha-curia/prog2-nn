import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import torch

ds_train=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
)

print(f'numbers of datasets: {len(ds_train)}')
#FashionMNISTは配列のようなものでできており一つ一つの要素が画像（image）とラベル（target）のタプルで構成されている
image,target=ds_train[0]
print(type(image),target)
plt.imshow(image,cmap="gray_r")
plt.title(target)
plt.show()

#PIL画像をtorch.Tensorに変換する 
image=transforms.functional.to_image(image)
image=transforms.functional.to_dtype(image,dtype=torch.float32,scale=True) #計算の時に値がでかくなりすぎないように0から1の少数に変換
print(image.shape,image.dtype)
print(image.min(),image.max())
