import torch
import torch.nn as nn
import torch.optim as optim
import os

import torchvision
import torchvision.transforms as transforms
from src.layers.s4 import S4Block as S4
from tqdm.auto import tqdm

dropout_fn = nn.Dropout2d

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(1, 784).t())]
)

train_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

val_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_set = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Dataloaders
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True, num_workers=12
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=128, shuffle=False, num_workers=12
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False, num_workers=12
)


class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


model = S4Model(
    d_input=1,
    d_output=10,
    d_model=64,
    n_layers=4,
    dropout=0.1,
)

model.to(device)


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s)
        for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(
            " | ".join(
                [
                    f"Optimizer group {i}",
                    f"{len(g['params'])} tensors",
                ]
                + [f"{k} {v}" for k, v in group_hps.items()]
            )
        )

    return optimizer, scheduler


criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(model, lr=0.01, weight_decay=0.01, epochs=50)


# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            "Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (
                batch_idx,
                len(train_loader),
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
                correct,
                total,
            )
        )


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                "Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    batch_idx,
                    len(dataloader),
                    eval_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                )
            )

    # Save checkpoint.
    if checkpoint:
        acc = 100.0 * correct / total
        if acc > best_acc:
            state = {
                "model": model.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/ckpt.pth")
            best_acc = acc

        return acc


pbar = tqdm(range(start_epoch, 50))
for epoch in pbar:
    val_acc = 0
    if epoch == 0:
        pbar.set_description("Epoch: %d" % (epoch))
    else:
        pbar.set_description("Epoch: %d | Val acc: %1.3f" % (epoch, val_acc))
    train()
    val_acc = eval(epoch, val_loader, checkpoint=True)
    eval(epoch, test_loader)
    scheduler.step()
