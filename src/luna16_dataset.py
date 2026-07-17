"""
LUNA16 Patch Dataset
====================
从 metadata.csv 读取索引，加载 (3, 64, 64) 的 .npy patch，
返回 PyTorch Tensor + 二元标签 (0=非结节, 1=结节)。

输入: data/processed/metadata.csv + patches/*.npy
输出: (B, 3, 64, 64) tensor, (B,) labels
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


class LUNA16Dataset(Dataset):
    def __init__(
        self,
        metadata_csv: str = "data/processed/metadata.csv",
        patches_dir: str = "data/processed/patches",
        split: Optional[str] = None,          # "train" / "val" / "test" / None=全部
        transform: Optional[callable] = None,  # 数据增强 (albumentations / torchvision)
    ):
        df = pd.read_csv(metadata_csv)

        # 修正 split 列 (兼容旧数据)
        if 'split' not in df.columns:
            df['split'] = ''
        df['split'] = df['split'].astype(str)
        df.loc[df['split'].isin(['nan', '']), 'split'] = ''
        if df['split'].nunique() == 1 and df['split'].iloc[0] == '':
            # auto-assign split if empty
            df.loc[df['subset_id'].isin(range(0, 8)), 'split'] = 'train'
            df.loc[df['subset_id'] == 8, 'split'] = 'val'
            df.loc[df['subset_id'] == 9, 'split'] = 'test'

        if split is not None:
            df = df[df['split'] == split].copy()

        self.df = df.reset_index(drop=True)
        self.patches_dir = Path(patches_dir)
        self.transform = transform

        n_pos = int(self.df['class'].sum())
        n_neg = len(self.df) - n_pos
        print(f"LUNA16Dataset (split={split or 'all'}): {len(self.df)} samples | "
              f"pos={n_pos} ({n_pos/max(1,len(self.df))*100:.1f}%) | "
              f"neg={n_neg} ({n_neg/max(1,len(self.df))*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[index]
        patch = np.load(str(self.patches_dir / row['patch_file']))  # (3, 64, 64) float32 [0,1]
        patch_tensor = torch.from_numpy(patch.copy()).float()  # copy to avoid negative stride issues

        # 如果 transform 存在 (torchvision)，直接应用在 C,H,W tensor 上
        if self.transform is not None:
            patch_tensor = self.transform(patch_tensor)

        label = int(row['class'])
        seriesuid = str(row['seriesuid'])
        return patch_tensor, label, seriesuid


def create_dataloaders(
    metadata_csv: str = "data/processed/metadata.csv",
    patches_dir: str = "data/processed/patches",
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """一键创建 train/val/test 三个 DataLoader"""
    train_ds = LUNA16Dataset(metadata_csv, patches_dir, split="train", transform=train_transform)
    val_ds = LUNA16Dataset(metadata_csv, patches_dir, split="val", transform=val_transform)
    test_ds = LUNA16Dataset(metadata_csv, patches_dir, split="test", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# 快速测试
if __name__ == "__main__":
    ds = LUNA16Dataset(split="train")
    print(f"Train length: {len(ds)}")
    x, y, uid = ds[0]
    print(f"Sample shape: {x.shape}, label: {y}, uid: {uid[:40]}...")
    # 测试 DataLoader
    loader, _, _ = create_dataloaders(batch_size=16, num_workers=0)
    xb, yb, _ = next(iter(loader))
    print(f"Batch shape: {xb.shape} (expect [16, 3, 64, 64]), labels: {yb[:8]}")
