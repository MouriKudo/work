"""Print environment info for screenshot"""
import torch, torchvision, numpy, pandas, SimpleITK, sklearn, matplotlib
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"numpy={numpy.__version__}")
print(f"pandas={pandas.__version__}")
print(f"SimpleITK={SimpleITK.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"matplotlib={matplotlib.__version__}")

# Directory structure
import os
from pathlib import Path
root = Path('D:/luna16-work')
print()
print("Project Structure:")
print("D:/luna16-work/")
print("  data/raw/")
print("    subset0/subset0/ - 89 CT scans (.mhd+.raw)")
print("    annotations.csv - 1186 nodules")
print("    candidates.csv - 551065 candidates")
print("    sampleSubmission.csv - Kaggle sample")
print("    data_manifest.csv - file MD5 checksums")
print("  data/processed/")
print("    patches/ - 444 patches (3x64x64)")
print("    metadata.csv - 444 rows x 13 cols")
print("  src/")
for f in sorted(Path('src').glob('*.py')):
    print(f"    {f.name}")
print("  paper_figs/")
for f in sorted(Path('paper_figs').glob('*')):
    print(f"    {f.name}")
print("  paper_code/ - original paper (read-only)")
print("  runs/ - training logs")
print("  requirements.txt")
