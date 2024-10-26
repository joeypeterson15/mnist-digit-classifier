import torch

weights = torch.ones(4, requires_grad=True)

print('weights:', weights)

model_output = (weights*3).sum()
model_output2 = model_output*2

model_output.backward()
# model_output2.backward()
print('model output', model_output)
print('model output2', model_output2)

print(weights.grad)
# for epoch in range(3):
