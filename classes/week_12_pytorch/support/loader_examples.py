"""Сборка батча в процессе DataLoader."""
import torch


def collate_points(examples):
    points, labels = zip(*examples)
    batch = {"points": torch.stack(points), "labels": torch.stack(labels)}
    return batch
