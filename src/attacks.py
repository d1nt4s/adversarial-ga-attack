"""
Baseline attack implementations for comparison with GA attack.
FGSM and PGD are white-box attacks used as benchmarks.
"""

import torch
import torch.nn as nn
import numpy as np

# CIFAR-10 normalization bounds: valid pixel range [0,1] mapped to normalized space
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
NORM_MIN = ((torch.zeros(3, 1, 1) - CIFAR_MEAN) / CIFAR_STD)  # normalized 0
NORM_MAX = ((torch.ones(3, 1, 1)  - CIFAR_MEAN) / CIFAR_STD)  # normalized 1


def fgsm_attack(model, image, label, epsilon=0.1):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2015).
    White-box: requires gradients.

    Returns: adversarial image (tensor)
    """
    norm_min = NORM_MIN.to(image.device)
    norm_max = NORM_MAX.to(image.device)

    image = image.clone().detach().requires_grad_(True)
    output = model(image)
    loss = nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * image.grad.sign()
    adv_image = torch.max(torch.min(image + perturbation, norm_max), norm_min).detach()
    return adv_image


def pgd_attack(model, image, label, epsilon=0.1, alpha=0.01, steps=40):
    """
    Projected Gradient Descent (Madry et al., 2018).
    White-box: requires gradients. Stronger than FGSM.

    Returns: adversarial image (tensor)
    """
    norm_min = NORM_MIN.to(image.device)
    norm_max = NORM_MAX.to(image.device)

    adv_image = image.clone().detach()
    adv_image = adv_image + torch.empty_like(adv_image).uniform_(-epsilon, epsilon)
    adv_image = torch.max(torch.min(adv_image, norm_max), norm_min).detach()

    for _ in range(steps):
        adv_image.requires_grad_(True)
        output = model(adv_image)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()

        adv_image = adv_image + alpha * adv_image.grad.sign()
        delta = torch.clamp(adv_image - image, -epsilon, epsilon)
        adv_image = torch.max(torch.min(image + delta, norm_max), norm_min).detach()

    return adv_image


def evaluate_attack(model, dataloader, attack_fn, device, max_samples=500):
    """
    Evaluate attack success rate on a dataloader.

    Returns: dict with accuracy and attack success rate
    """
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0

    for images, labels in dataloader:
        if total >= max_samples:
            break

        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)
            correct_clean += (clean_preds == labels).sum().item()

        adv_images = attack_fn(model, images, labels)
        with torch.no_grad():
            adv_preds = model(adv_images).argmax(dim=1)
            correct_adv += (adv_preds == labels).sum().item()

        total += images.size(0)

    clean_acc = correct_clean / total
    robust_acc = correct_adv / total
    attack_success_rate = 1 - robust_acc

    return {
        "clean_accuracy": clean_acc,
        "robust_accuracy": robust_acc,
        "attack_success_rate": attack_success_rate,
        "total_samples": total,
    }
