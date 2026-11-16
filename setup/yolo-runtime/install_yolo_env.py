"""Create a separate CPU YOLO runtime using an existing Conda installation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
RESULT_MARKER = 'YOLO_SETUP_RESULT='
VERIFY_CODE = '; '.join(line.strip() for line in '''import json, sys
import torch, torchvision, ultralytics, onnx, onnxruntime, onnxslim
from ultralytics import YOLO
boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]])
assert torchvision.ops.nms(boxes, torch.tensor([.9, .8]), .5).tolist() == [0]
YOLO("yolo26n.yaml", verbose=False)
print("YOLO_SETUP_RESULT=" + json.dumps({"python": sys.executable, "torch": torch.__version__, "ultralytics": ultralytics.__version__, "cudaAvailable": torch.cuda.is_available()}))
'''.splitlines() if line.strip())


def run(command, capture=False):
    if not capture:
        print('\n> ' + subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment['PYTHONIOENCODING'] = 'utf-8'
    return subprocess.run(command, check=True, capture_output=capture, text=True,
                          encoding='utf-8', errors='replace', env=environment)


def find_conda(configured=None):
    candidate = configured or os.environ.get('CONDA_EXE') or shutil.which('conda.exe') or shutil.which('conda')
    if not candidate or not Path(candidate).is_file():
        raise ValueError('未找到 Conda。请打开 Anaconda Prompt，在安装脚本目录重新运行。')
    if Path(candidate).suffix.lower() in ('.bat', '.cmd'):
        raise ValueError('请通过 --conda 指定 Anaconda 安装目录下 Scripts/conda.exe，而不是 conda.bat。')
    return str(Path(candidate).resolve())


def verify(conda, name):
    result = run([conda, 'run', '--no-capture-output', '-n', name, 'python', '-c', VERIFY_CODE], capture=True)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            info = json.loads(line[len(RESULT_MARKER):])
            if not Path(info['python']).is_file():
                break
            return info
    raise ValueError('环境检查没有返回有效 Python 路径，请检查上方输出。')


def install(conda, name='yolo26', repair=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', name) or name.casefold() in {'base', 'root'}:
        raise ValueError('环境名只能包含字母、数字、横线和下划线，且不能使用 base 或 root。')
    requirements = ROOT / 'requirements-yolo.txt'
    if not requirements.is_file():
        raise ValueError('缺少 requirements-yolo.txt，请将整个安装包解压后再运行，不要只复制 bat 文件。')
    listing = json.loads(run([conda, 'env', 'list', '--json'], capture=True).stdout)
    exists = any(Path(value).name.casefold() == name.casefold() for value in listing['envs'])
    if exists and not repair:
        print(f'发现已有 {name} 环境，先检查，不修改其依赖。', flush=True)
        try:
            return verify(conda, name)
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise ValueError(f'已有 {name} 环境未通过检查，已保留不变。可以用 --env yolo26_platform 新建独立环境；'
                             '确认允许更新已有环境依赖时才使用 --repair。') from exc
    if not exists:
        run([conda, 'create', '-y', '-n', name, 'python=3.11', 'pip'])
    else:
        print(f'--repair 已指定：将更新 {name} 环境的依赖；不会删除环境。', flush=True)
    prefix = [conda, 'run', '--no-capture-output', '-n', name, 'python', '-m', 'pip']
    run([*prefix, 'install', '--upgrade', 'pip'])
    run([*prefix, 'install', '--upgrade', 'torch>=2.2,<3', 'torchvision>=0.17,<1', '--index-url', CPU_INDEX])
    run([*prefix, 'install', '-r', str(requirements)])
    run([*prefix, 'check'])
    return verify(conda, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conda', help='Path to conda.exe')
    parser.add_argument('--env', default='yolo26', help='Separate environment name, default yolo26')
    parser.add_argument('--repair', action='store_true', help='Allow package updates in an existing environment')
    args = parser.parse_args()
    try:
        info = install(find_conda(args.conda), args.env, args.repair)
        print('\n环境检查通过！')
        print(f"PyTorch: {info['torch']}  Ultralytics: {info['ultralytics']}")
        print('请复制下面的路径，填入平台“本机设置 → AI Python 路径”，或自动标注页的 Python 路径：')
        print(info['python'])
        print('默认使用 CPU。自动标注的设备请先填 cpu；GPU 版需按显卡与驱动另行选择。')
        print('未修改平台设置或系统环境变量，也未下载模型权重。')
        try:
            (ROOT / 'yolo_python_path.txt').write_text(info['python'] + '\n', encoding='utf-8')
            print('路径也已保存到同目录 yolo_python_path.txt。')
        except OSError:
            print('安装目录不可写，请手动复制上方路径。')
        return 0
    except subprocess.CalledProcessError as exc:
        print(f'\n安装或检查失败（退出码 {exc.returncode}）。', file=sys.stderr)
        if exc.stdout: print(exc.stdout[-4000:], file=sys.stderr)
        if exc.stderr: print(exc.stderr[-4000:], file=sys.stderr)
        print('请检查网络／磁盘空间以及 Conda 或 pip 的错误提示。若是仓库条款提示，请先在 Anaconda Prompt 中自行审阅处理。'
              '不会自动接受条款、关闭证书校验或删除环境；修复后可用 --repair 重试。', file=sys.stderr)
        return 1
    except (ValueError, OSError, KeyError) as exc:
        print(f'\n{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
