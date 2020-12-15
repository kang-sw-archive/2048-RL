# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (3, 3))
        self.conv2 = nn.Conv2d(6, 16, (3, 3))
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor):
        # (2, 2) 크기 윈도우에 대해 맥스 풀링(max pooling)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 제곱수라면 하나의 숫자만을 특정
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x: Tensor):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# %%
input = torch.randn(1, 1, 32, 32)
out: Tensor = net(input)
print(out)

# %%
net.zero_grad()
out.backward(torch.randn(1, 10))

# %%
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# %%
n = loss.grad_fn


def recursive_show(arg, lv=0):
    for j in arg.next_functions:
        print('  ' * lv + str(j))
        if j[0] is not None:
            recursive_show(j[0], lv + 1)


recursive_show(n)


# %%
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# %%
