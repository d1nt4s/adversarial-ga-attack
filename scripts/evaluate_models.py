"""
Оценка baseline и robust моделей против FGSM, PGD, GA.
Результаты сохраняются в results/tables/evaluation.json
"""
import sys
sys.path.append('.')

import json
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from src.model import CNN
from src.attacks import fgsm_attack, pgd_attack
from src.ga import GeneticAttack, NORM_MIN, NORM_MAX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

rng = np.random.default_rng(42)
all_indices = rng.permutation(len(testset))
idx_1000 = all_indices[:1000].tolist()
idx_100  = all_indices[:100].tolist()

loader_1000 = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, idx_1000), batch_size=128, shuffle=False)
loader_100  = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, idx_100),  batch_size=1,   shuffle=False)

baseline = CNN().to(device)
baseline.load_state_dict(torch.load('results/model.pth', map_location=device))
baseline.eval()

robust = CNN().to(device)
robust.load_state_dict(torch.load('results/model_robust.pth', map_location=device))
robust.eval()

print('Модели загружены.\n')

EPSILON_GRAD = 0.03   # для FGSM/PGD — стандарт литературы
EPSILON_GA   = 0.10   # для GA — наш лучший результат


def clean_accuracy(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total


def grad_attack_asr(model, loader, attack_fn):
    """ASR считается только по правильно классифицированным картинкам."""
    successes, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            correct_mask = model(images).argmax(dim=1) == labels
        if correct_mask.sum() == 0:
            continue
        imgs, lbls = images[correct_mask], labels[correct_mask]
        adv = attack_fn(model, imgs, lbls)
        with torch.no_grad():
            successes += (model(adv).argmax(dim=1) != lbls).sum().item()
        total += len(lbls)
    return successes / total if total > 0 else 0.0


def ga_asr(model, loader):
    ga = GeneticAttack(model, population_size=50, generations=30,
                       mutation_rate=0.1, mutation_strength=0.05,
                       epsilon=EPSILON_GA, device=str(device))
    successes, total = 0, 0
    for i, (image, label) in enumerate(loader):
        image_np  = image[0].numpy()
        true_label = label[0].item()
        with torch.no_grad():
            if model(image.to(device)).argmax(dim=1).item() != true_label:
                continue  # пропускаем уже неправильные
        best_pert, _ = ga.attack(image_np, true_label)
        adv_np = np.clip(image_np + best_pert, NORM_MIN, NORM_MAX)
        adv_t  = torch.FloatTensor(adv_np).unsqueeze(0).to(device)
        with torch.no_grad():
            adv_pred = model(adv_t).argmax(dim=1).item()
        if adv_pred != true_label:
            successes += 1
        total += 1
        if (i + 1) % 10 == 0:
            print(f'  GA: {i+1}/100 | ASR so far: {successes/total:.3f}')
    return successes / total if total > 0 else 0.0


results = {}

for name, model in [('baseline', baseline), ('robust', robust)]:
    print(f'=== {name.upper()} MODEL ===')

    ca = clean_accuracy(model, loader_1000)
    print(f'  Clean Acc:  {ca:.4f}')

    fgsm_asr = grad_attack_asr(model, loader_1000,
                               lambda m, x, y: fgsm_attack(m, x, y, epsilon=EPSILON_GRAD))
    print(f'  FGSM ASR:   {fgsm_asr:.4f}  (eps={EPSILON_GRAD})')

    pgd_asr = grad_attack_asr(model, loader_1000,
                              lambda m, x, y: pgd_attack(m, x, y, epsilon=EPSILON_GRAD, steps=40))
    print(f'  PGD ASR:    {pgd_asr:.4f}  (eps={EPSILON_GRAD})')

    print(f'  GA ASR (N=100, eps={EPSILON_GA})...')
    g_asr = ga_asr(model, loader_100)
    print(f'  GA ASR:     {g_asr:.4f}')

    results[name] = {
        'clean_acc': round(ca, 4),
        'fgsm_asr':  round(fgsm_asr, 4),
        'pgd_asr':   round(pgd_asr, 4),
        'ga_asr':    round(g_asr, 4),
    }
    print()

os.makedirs('results/tables', exist_ok=True)
with open('results/tables/evaluation.json', 'w') as f:
    json.dump(results, f, indent=2)

print('=== ИТОГОВАЯ ТАБЛИЦА ===')
print(f"{'Модель':<12} {'Clean Acc':>10} {'FGSM ASR':>10} {'PGD ASR':>10} {'GA ASR':>10}")
print('-' * 52)
for name, r in results.items():
    print(f"{name:<12} {r['clean_acc']:>10.4f} {r['fgsm_asr']:>10.4f} {r['pgd_asr']:>10.4f} {r['ga_asr']:>10.4f}")

print('\nСохранено в results/tables/evaluation.json')
