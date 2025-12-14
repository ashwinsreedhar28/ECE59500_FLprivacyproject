import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.fc = nn.Linear(32 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PrivacyConfig:
    def __init__(
        self,
        use_secure_agg=False,
        use_local_dp=False,
        use_adaptive_clip=False,
        dp_noise_std=0.05,
        base_clip_norm=1.0,
        clip_momentum=0.9,
        secure_agg_group_size=3,
    ):
        self.use_secure_agg = use_secure_agg
        self.use_local_dp = use_local_dp
        self.use_adaptive_clip = use_adaptive_clip
        self.dp_noise_std = dp_noise_std
        self.base_clip_norm = base_clip_norm
        self.clip_momentum = clip_momentum
        self.secure_agg_group_size = secure_agg_group_size
        
def tv_loss(x):
    return ((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2).mean() + \
           ((x[:, :, :, :-1] - x[:, :, :, 1:]) ** 2).mean()


def apply_privacy_to_grads(grads, client, privacy_cfg: PrivacyConfig):
    priv_grads = [g.clone() for g in grads]

    if privacy_cfg.use_adaptive_clip:
        flat = torch.cat([g.view(-1) for g in priv_grads])
        norm = flat.norm() + 1e-8

        if client.running_grad_norm is None:
            client.running_grad_norm = norm.detach()
        else:
            client.running_grad_norm = (
                privacy_cfg.clip_momentum * client.running_grad_norm
                + (1.0 - privacy_cfg.clip_momentum) * norm.detach()
            )

        clip_norm = privacy_cfg.base_clip_norm * float(client.running_grad_norm)

        if norm > clip_norm:
            scale = clip_norm / norm
            priv_grads = [g * scale for g in priv_grads]

    if privacy_cfg.use_local_dp and privacy_cfg.dp_noise_std > 0.0:
        priv_grads = [
            g + privacy_cfg.dp_noise_std * torch.randn_like(g)
            for g in priv_grads
        ]

    return priv_grads

def get_mnist_dataloaders(num_clients=5, batch_size=32, iid=True):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
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


def get_mnist_test_loader(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Client:
    def __init__(self, client_id, dataloader, device="cpu"):
        self.id = client_id
        self.dataloader = dataloader
        self.device = device
        self.running_grad_norm = None  # for adaptive clipping

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

    def get_one_batch_and_gradients(self, model, max_batch_size=1, privacy_cfg: PrivacyConfig = None):
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

        if privacy_cfg is not None:
            grads = apply_privacy_to_grads(grads, self, privacy_cfg)

        return x.detach().clone(), y.detach().clone(), grads


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

def dlg_attack_known_label(
    model,
    shared_grads,
    img_shape,
    labels,
    iters=800,
    lr=0.05,
    device="cpu",
    print_every=100,
    use_tv=True,
    tv_weight=1e-4,
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

        def normalize(g):
            return g / (g.norm() + 1e-8)

        grad_diff = 0
        for g_fake, g_real in zip(grads_fake, shared_grads):
            grad_diff += F.mse_loss(normalize(g_fake), normalize(g_real))

        total_obj = grad_diff
        if use_tv:
            total_obj = total_obj + tv_weight * tv_loss(dummy_x)

        total_obj.backward()
        optimizer.step()

        with torch.no_grad():
            dummy_x.clamp_(0, 1)

        if (it + 1) % print_every == 0:
            print(f"[DLG-MNIST-known-label] Iter {it+1}/{iters}, grad_diff={grad_diff.item():.6f}")

    return dummy_x.detach().cpu()

def evaluate_accuracy(model, dataloader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def evaluate_reconstruction_accuracy(model, recon_images, true_labels, device="cpu"):
    model.eval()
    recon_images = recon_images.to(device)
    true_labels = true_labels.to(device)
    with torch.no_grad():
        logits = model(recon_images)
        preds = logits.argmax(dim=1)
        correct = (preds == true_labels).sum().item()
        total = true_labels.size(0)
    return correct / total

def run_experiment_rounds_mnist(
    num_rounds=5,
    num_clients=5,
    num_attacks_per_round=5,
    batch_size=32,
    dlg_batch_size=1,
    dlg_iters=800,
    device="cpu",
    privacy_cfg: PrivacyConfig = None,
):
    client_loaders = get_mnist_dataloaders(
        num_clients=num_clients,
        batch_size=batch_size,
        iid=True,
    )
    test_loader = get_mnist_test_loader(batch_size=256)

    global_model = SimpleCNN().to(device)
    server = Server(global_model, device=device)

    clients = [
        Client(i, client_loaders[i], device=device)
        for i in range(num_clients)
    ]

    train_accs = []
    recon_accs = []

    for rnd in range(num_rounds):
        print(f"\n=== Federated Round {rnd+1}/{num_rounds} ===")

        client_states = []
        client_sizes = []
        for client in clients:
            num_samples = len(client.dataloader.dataset)
            client_sizes.append(num_samples)
            updated_state = client.local_train_one_epoch(global_model, lr=0.01)
            client_states.append(updated_state)

        server.aggregate(client_states, client_sizes)
        global_model = server.global_model

        train_acc = evaluate_accuracy(global_model, test_loader, device=device)
        print(f"[Round {rnd+1}] Global test accuracy: {train_acc:.4f}")

        x_recon_all = []
        y_recon_all = []

        for k in range(num_attacks_per_round):
            victim_idx = np.random.randint(0, num_clients)
            victim_client = clients[victim_idx]

            if privacy_cfg is not None and privacy_cfg.use_secure_agg:
                group_size = min(privacy_cfg.secure_agg_group_size, num_clients)
                group_indices = set([victim_idx])
                all_indices = list(range(num_clients))
                remaining = [i for i in all_indices if i != victim_idx]
                if len(remaining) > 0 and group_size > 1:
                    extra = random.sample(remaining, k=group_size - 1)
                    group_indices.update(extra)
                group_indices = sorted(list(group_indices))

                grads_group = []
                x_real, y_real = None, None

                for idx in group_indices:
                    c = clients[idx]
                    x_i, y_i, g_i = c.get_one_batch_and_gradients(
                        global_model,
                        max_batch_size=dlg_batch_size,
                        privacy_cfg=privacy_cfg,
                    )
                    grads_group.append(g_i)
                    if idx == victim_idx:
                        x_real, y_real = x_i, y_i

                attack_grads = []
                num_group = len(grads_group)
                for layer_idx in range(len(grads_group[0])):
                    s = sum(g[layer_idx] for g in grads_group) / float(num_group)
                    attack_grads.append(s.detach())
            else:
                x_real, y_real, attack_grads = victim_client.get_one_batch_and_gradients(
                    global_model,
                    max_batch_size=dlg_batch_size,
                    privacy_cfg=privacy_cfg,
                )

            dummy_x = dlg_attack_known_label(
                model=global_model,
                shared_grads=attack_grads,
                img_shape=x_real.shape,
                labels=y_real,
                iters=dlg_iters,
                lr=0.05,
                device=device,
                print_every=dlg_iters + 1, 
                use_tv=True,
                tv_weight=1e-4,
            )

            x_recon_all.append(dummy_x[0])   
            y_recon_all.append(y_real[0])

        x_recon_all = torch.stack(x_recon_all)  
        y_recon_all = torch.stack(y_recon_all)  

        recon_acc = evaluate_reconstruction_accuracy(
            global_model,
            x_recon_all,
            y_recon_all,
            device=device,
        )
        print(f"[Round {rnd+1}] Reconstruction accuracy: {recon_acc:.4f}")

        train_accs.append(train_acc)
        recon_accs.append(recon_acc)

    return train_accs, recon_accs

def plot_all_experiments(results, title):
    rounds = np.arange(1, len(next(iter(results.values()))["train"]) + 1)

    plt.figure(figsize=(8, 5))

    for name, res in results.items():
        plt.plot(rounds, res["train"], marker="o", linestyle="-",
                 label=f"{name} – Test Acc")
        plt.plot(rounds, res["recon"], marker="x", linestyle="--",
                 label=f"{name} – Recon Acc")

    plt.xlabel("FL Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def get_experiment_configs(base_args):
    return {
        "Baseline": PrivacyConfig(),

        "Adaptive Clip": PrivacyConfig(
            use_adaptive_clip=True,
            base_clip_norm=base_args.base_clip_norm,
            clip_momentum=base_args.clip_momentum,
        ),

        "Local DP": PrivacyConfig(
            use_local_dp=True,
            dp_noise_std=base_args.dp_noise_std,
        ),

        "Secure Aggregation": PrivacyConfig(
            use_secure_agg=True,
            secure_agg_group_size=base_args.secure_agg_group_size,
        ),

        "Clip + LDP": PrivacyConfig(
            use_adaptive_clip=True,
            use_local_dp=True,
            dp_noise_std=base_args.dp_noise_std,
            base_clip_norm=base_args.base_clip_norm,
            clip_momentum=base_args.clip_momentum,
        ),

        "Clip + SecAgg": PrivacyConfig(
            use_adaptive_clip=True,
            use_secure_agg=True,
            base_clip_norm=base_args.base_clip_norm,
            clip_momentum=base_args.clip_momentum,
            secure_agg_group_size=base_args.secure_agg_group_size,
        ),

        "LDP + SecAgg": PrivacyConfig(
            use_local_dp=True,
            use_secure_agg=True,
            dp_noise_std=base_args.dp_noise_std,
            secure_agg_group_size=base_args.secure_agg_group_size,
        ),

        "Clip + LDP + SecAgg": PrivacyConfig(
            use_adaptive_clip=True,
            use_local_dp=True,
            use_secure_agg=True,
            dp_noise_std=base_args.dp_noise_std,
            base_clip_norm=base_args.base_clip_norm,
            clip_momentum=base_args.clip_momentum,
            secure_agg_group_size=base_args.secure_agg_group_size,
        ),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST FL + DLG privacy comparison")
    parser.add_argument("--dp_noise_std", type=float, default=0.05)
    parser.add_argument("--base_clip_norm", type=float, default=1.0)
    parser.add_argument("--clip_momentum", type=float, default=0.9)
    parser.add_argument("--secure_agg_group_size", type=int, default=3)
    parser.add_argument("--dlg_iters", type=int, default=800)
    parser.add_argument("--num_attacks_per_round", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    experiment_cfgs = get_experiment_configs(args)

    all_results = {}

    for name, privacy_cfg in experiment_cfgs.items():
        print(f"\n==============================")
        print(f"Running experiment: {name}")
        print(f"==============================")

        train_accs, recon_accs = run_experiment_rounds_mnist(
            num_rounds=args.num_rounds,
            num_clients=5,
            num_attacks_per_round=args.num_attacks_per_round,
            batch_size=32,
            dlg_batch_size=1,
            dlg_iters=args.dlg_iters,
            device=device,
            privacy_cfg=privacy_cfg,
        )

        all_results[name] = {
            "train": train_accs,
            "recon": recon_accs,
        }

    plot_all_experiments(
        all_results,
        title="MNIST – Utility vs Leakage Across Privacy Mechanisms",
    )
