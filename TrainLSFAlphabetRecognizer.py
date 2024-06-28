import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

from src.alphabet_recognizer import LSFAlphabetRecognizer

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_tensor = torch.tensor(item['input'], dtype=torch.float32)
        label_tensor = torch.tensor(item['label_id'], dtype=torch.long)
        return input_tensor, label_tensor

dataset = CustomDataset(load_data("alphabet_dataset.json"))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Example data

# Initialize the networks
model = LSFAlphabetRecognizer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(outputs)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model.pth')
