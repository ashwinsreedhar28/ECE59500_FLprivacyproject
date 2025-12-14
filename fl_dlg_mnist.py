import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
import random

# CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.fc = nn.Linear(32 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# Dataloaders
def get_mnist_dataloaders(num_clients=5, batch_size=32, iid=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    num_samples = len(train_dataset)
    indices = np.arange(num_samples)

    if iid:
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)
    else:
        labels = np.array(train_dataset.targets)
        sorted_indices = indices[np.argsort(labels)]
        split_indices = np.array_split(sorted_indices, num_clients)

    client_loaders = []
    for idxs in split_indices:
        subset = Subset(train_dataset, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    return client_loaders

# Client for FL system
class Client:
    def __init__(self, client_id, dataloader, device="cpu"):
        self.id = client_id
        self.dataloader = dataloader
        self.device = device

    def local_train_one_epoch(self, model, lr=0.01):
        model = copy.deepcopy(model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        return model.state_dict()

    def get_one_batch_and_gradients(self, model, lr=0.01, max_batch_size=1):
        model = copy.deepcopy(model).to(self.device)
        model.train()
        criterion = nn.CrossEntropyLoss()

        x, y = next(iter(self.dataloader))
        if max_batch_size is not None:
            x = x[:max_batch_size]
            y = y[:max_batch_size]

        x, y = x.to(self.device), y.to(self.device)

        model.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

        return x.detach().clone(), y.detach().clone(), grads

# Server for FL system
class Server:
    def __init__(self, global_model, device="cpu"):
        self.global_model = global_model
        self.device = device

    def aggregate(self, client_states, client_sizes):
        total_samples = float(sum(client_sizes))
        new_state = copy.deepcopy(client_states[0])
        for key in new_state.keys():
            new_state[key] = new_state[key] * (client_sizes[0] / total_samples)

        for i in range(1, len(client_states)):
            state = client_states[i]
            weight = client_sizes[i] / total_samples
            for key in new_state.keys():
                new_state[key] += state[key] * weight

        self.global_model.load_state_dict(new_state)

def compute_grads_from_data(model, x, y, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]
    return grads

# DLG attack
def dlg_attack(model, shared_grads, img_shape, num_classes=10,
               iters=800, lr=0.1, device="cpu", print_every=50):
    model = copy.deepcopy(model).to(device)
    model.eval()

    batch_size, C, H, W = img_shape

    dummy_x = torch.randn((batch_size, C, H, W), device=device, requires_grad=True)
    dummy_y_logits = torch.randn((batch_size, num_classes), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([dummy_x, dummy_y_logits], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for it in range(iters):
        optimizer.zero_grad()

        dummy_y_soft = F.softmax(dummy_y_logits, dim=-1)
        
        pred = model(dummy_x)
        log_probs = F.log_softmax(pred, dim=-1)
        loss_label = -(dummy_y_soft * log_probs).sum(dim=1).mean()

        grads_fake = torch.autograd.grad(
            loss_label,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=True
        )

        grad_diff = 0
        for g_fake, g_real in zip(grads_fake, shared_grads):
            grad_diff += F.mse_loss(g_fake, g_real)

        total_loss = grad_diff

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            dummy_x.clamp_(0, 1)

        if (it + 1) % print_every == 0:
            print(f"[DLG] Iter {it+1}/{iters}, grad_diff={grad_diff.item():.6f}")

    return dummy_x.detach().cpu(), dummy_y_logits.detach().cpu()

# Known-label DLG
def dlg_attack_known_label(
    model,
    shared_grads,
    img_shape,
    labels,
    iters=800,
    lr=0.02,
    device="cpu",
    print_every=100,
):
    model = copy.deepcopy(model).to(device)
    model.eval()

    labels = labels.to(device)
    batch_size, C, H, W = img_shape

    dummy_x = torch.randn((batch_size, C, H, W), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_x], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for it in range(iters):
        optimizer.zero_grad()

        pred = model(dummy_x)
        loss = criterion(pred, labels)

        grads_fake = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=True
        )

        grad_diff = 0
        for g_fake, g_real in zip(grads_fake, shared_grads):
            grad_diff += F.mse_loss(g_fake, g_real)

        grad_diff.backward()
        optimizer.step()

        with torch.no_grad():
            dummy_x.clamp_(0, 1)

        if (it + 1) % print_every == 0:
            print(f"[DLG-known-label] Iter {it+1}/{iters}, grad_diff={grad_diff.item():.6f}")

    return dummy_x.detach().cpu()

# 1 round of DLG
def simulate_one_fl_round_with_dlg(
    num_clients=5,
    victim_client_idx=0,
    batch_size=32,
    dlg_batch_size=1,
    dlg_iters=800,
    device="cpu"
):
    client_loaders = get_mnist_dataloaders(num_clients=num_clients,
                                           batch_size=batch_size,
                                           iid=True)

    global_model = SimpleCNN().to(device)
    server = Server(global_model, device=device)

    clients = [
        Client(i, client_loaders[i], device=device)
        for i in range(num_clients)
    ]

    client_states = []
    client_sizes = []

    for i, client in enumerate(clients):
        num_samples = len(client.dataloader.dataset)
        client_sizes.append(num_samples)

        updated_state = client.local_train_one_epoch(global_model, lr=0.01)
        client_states.append(updated_state)

    server.aggregate(client_states, client_sizes)
    print("FedAvg aggregation complete. Global model updated.")

    victim_client = clients[victim_client_idx]
    print(f"Running attack on client {victim_client_idx}...")

    x_real, y_real, grads_real = victim_client.get_one_batch_and_gradients(
        server.global_model,
        lr=0.01,
        max_batch_size=dlg_batch_size
    )

    print(f"Victim batch shape: {x_real.shape}, labels: {y_real}")

    dummy_x = dlg_attack_known_label(
        model=server.global_model,
        shared_grads=grads_real,
        img_shape=x_real.shape,
        labels=y_real,
        iters=dlg_iters,
        lr=0.02,
        device=device,
        print_every=100,
    )

    dummy_labels = y_real.clone() 

    print(f"True labels:   {y_real.cpu().numpy()}")
    print(f"Recovered lbl: {dummy_labels.numpy()}")

    return x_real.cpu(), y_real.cpu(), dummy_x, dummy_labels

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    x_real, y_real, x_recon, y_recon = simulate_one_fl_round_with_dlg(
        num_clients=5,
        victim_client_idx=0,
        batch_size=32,
        dlg_batch_size=1,
        dlg_iters=800,
        device=device
    )

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(x_real[0, 0], cmap="gray")
        axes[0].set_title(f"Real (label={y_real[0].item()})")
        axes[0].axis("off")

        axes[1].imshow(x_recon[0, 0], cmap="gray")
        axes[1].set_title(f"Recon (label={y_recon[0].item()})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed; skipping visualization.")
