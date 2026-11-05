#!/usr/bin/env python3
"""Report the actual execution environment without starting model computation."""
import importlib.metadata
import json
import platform
import sys


def main():
    packages = ("numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "torch", "easydict", "tqdm")
    report = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "machine": platform.machine(), "packages": {}}
    missing = []
    for package in packages:
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
            missing.append(package)
    if "torch" not in missing:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    report["missing_packages"] = missing
    print(json.dumps(report, indent=2))
    return bool(missing)


if __name__ == "__main__":
    raise SystemExit(main())
