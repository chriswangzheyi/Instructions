"""下载 Qwen3-0.6B → models/Qwen3-0.6B（约 1.2GB）。

默认先 ModelScope（国内稳），失败再试 HF 镜像。
与 LoRA 学习无关，只是把基座权重弄到手。

用法::

    /opt/anaconda3/bin/python3 download_qwen.py
    /opt/anaconda3/bin/python3 download_qwen.py --source modelscope
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "models" / "Qwen3-0.6B"
HF_REPO = "Qwen/Qwen3-0.6B"
MS_REPO = "Qwen/Qwen3-0.6B"


def looks_complete(path: Path) -> bool:
    if not path.exists():
        return False
    has_config = (path / "config.json").exists()
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return has_config and has_weights


def download_hf(target: Path) -> Path:
    # 镜像 + 关掉 Xet，否则流量会绕开镜像打到 huggingface CDN。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}  HF_HUB_DISABLE_XET=1")
    return Path(
        snapshot_download(
            HF_REPO,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    )


def ensure_modelscope() -> None:
    try:
        import modelscope  # noqa: F401
        return
    except ImportError:
        pass
    print("未安装 modelscope，正在 pip install ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "modelscope"])


def download_modelscope(target: Path) -> Path:
    ensure_modelscope()
    from modelscope import snapshot_download

    # modelscope 会写到 cache；指定 local_dir 后文件落在目标目录。
    path = snapshot_download(MS_REPO, local_dir=str(target))
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Qwen3-0.6B")
    parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "hf"),
        default="auto",
        help="auto=先 ModelScope，失败再试 HF 镜像",
    )
    parser.add_argument("--force", action="store_true", help="目录已有模型也重新下")
    args = parser.parse_args()

    TARGET.mkdir(parents=True, exist_ok=True)
    if looks_complete(TARGET) and not args.force:
        print(f"已存在完整模型: {TARGET}")
        print("若要重下: python download_qwen.py --force")
        return

    print(f"下载到 {TARGET}")
    errors: list[str] = []

    order = {
        "auto": ("modelscope", "hf"),
        "modelscope": ("modelscope",),
        "hf": ("hf",),
    }[args.source]

    for source in order:
        try:
            print(f"\n>>> 尝试来源: {source}")
            path = download_modelscope(TARGET) if source == "modelscope" else download_hf(TARGET)
            if not looks_complete(TARGET):
                raise RuntimeError(f"{source} 下完后仍缺 config/权重，目录={path}")
            print("完成:", TARGET)
            return
        except Exception as exc:
            msg = f"{source} 失败: {type(exc).__name__}: {exc}"
            print(msg)
            errors.append(msg)

    print("\n全部来源都失败了。可手动试:")
    print("  1) /opt/anaconda3/bin/pip install -U modelscope")
    print("  2) /opt/anaconda3/bin/python3 download_qwen.py --source modelscope")
    print("  3) 浏览器打开 https://www.modelscope.cn/models/Qwen/Qwen3-0.6B 下载后解压到:")
    print(f"     {TARGET}")
    raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
