"""Directory-isolated dedup batches with an exact, persistent rollback scope."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .handlers import RunContext
from .task_manager import TaskCancelled


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _folders(root: Path, context: RunContext) -> list[Path]:
    from .image_dedup import BACKUP_FOLDER, IMAGE_SUFFIXES, _natural_key

    folders = []
    def scan_error(error):
        raise ValueError(f'无法扫描目录，尚未执行去重：{error.filename}（{error.strerror}）')

    for current, directories, files in os.walk(root, topdown=True, followlinks=False, onerror=scan_error):
        context.check_cancelled()
        folder = Path(current)
        directories[:] = sorted((name for name in directories
                                 if not name.endswith(BACKUP_FOLDER) and name != '.git'
                                 and not (folder / name).is_symlink()), key=lambda name: _natural_key(Path(name)))
        if any(Path(name).suffix.lower() in IMAGE_SUFFIXES and not (folder / name).is_symlink() for name in files):
            folders.append(folder)
    return folders


def _child_folder(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise RuntimeError('批量去重清单包含不安全的目录，已停止回滚。')
    folder = root / path
    if folder.resolve() != folder or not folder.is_dir():
        raise RuntimeError(f'图片目录已移动、缺失或变成链接，已停止回滚：{folder}')
    return folder


def _active_groups(root: Path, batch: dict) -> list[tuple]:
    from .image_dedup import _backup_root, _rollback_moves

    groups = []
    for result in batch.get('folderResults', []):
        if not result.get('backupManifest'):
            continue
        folder = _child_folder(root, result['relativeFolder'])
        manifest_path = Path(result['backupManifest'])
        expected = _backup_root(folder)
        if (manifest_path.name != 'manifest.json' or manifest_path.parent.parent != expected
                or manifest_path.resolve() != manifest_path):
            raise RuntimeError('批量备份清单路径不匹配，已停止回滚。')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status') == 'rolled_back':
            continue  # May already have been restored through the single-folder UI.
        moves = _rollback_moves(folder, manifest_path, manifest)
        groups.append((manifest_path, manifest, moves))
    return groups


def _rollback_batch(root: Path, context: RunContext) -> dict:
    from .image_dedup import IMAGE_SUFFIXES, _backup_root

    # Do not scan child directories for their latest unrelated standalone jobs.
    # Only the exact manifests belonging to this root's latest active batch count.
    for path in sorted(_backup_root(root).glob('*/batch.json'), reverse=True):
        if path.is_symlink() or path.resolve() != path:
            continue
        batch = json.loads(path.read_text(encoding='utf-8'))
        if (batch.get('sourceFolder') != str(root) or batch.get('preview')
                or batch.get('status') == 'rolled_back'):
            continue
        groups = _active_groups(root, batch)  # All conflicts checked before restoring anything.
        if not groups:
            continue
        context.check_cancelled()
        completed = []
        originals = [(manifest_path, manifest_path.read_bytes()) for manifest_path, _, _ in groups]
        original_batch = path.read_bytes()
        try:
            for manifest_path, manifest, moves in groups:
                for source, destination in moves:
                    context.check_cancelled()
                    source.replace(destination)
                    completed.append((source, destination))
                context.report(f'回滚检查通过，已恢复目录：{manifest["sourceFolder"]}')
            for manifest_path, manifest, _ in groups:
                _write(manifest_path, {**manifest, 'status': 'rolled_back',
                                      'rolledBackAt': datetime.now().isoformat(timespec='seconds')})
            _write(path, {**batch, 'status': 'rolled_back',
                          'rolledBackAt': datetime.now().isoformat(timespec='seconds')})
        except Exception:
            for source, destination in reversed(completed):
                if destination.exists() and not source.exists():
                    destination.replace(source)
            for manifest_path, original in originals:
                manifest_path.write_bytes(original)
            path.write_bytes(original_batch)
            raise
        restored = sum(destination.suffix.lower() in IMAGE_SUFFIXES for _, destination in completed)
        message = f'批量去重已回滚：恢复 {len(groups)} 个目录、{restored} 张图片及对应标注。'
        context.report(message)
        return {'message': message, 'restoredCount': restored, 'folderCount': len(groups),
                'batchReport': str(path), 'outputFiles': [str(path)], 'outputFolders': [str(root)]}
    raise ValueError('该上层目录没有可以回滚的批量去重记录；单目录记录请关闭“包含子文件夹”并选择对应图片目录。')


def run_batch(context: RunContext) -> dict:
    from .image_dedup import _backup_root, _run_single_folder

    raw = context.paths.get('images_folder')
    if not raw or not raw.expanduser().is_dir():
        raise ValueError('请选择存在的图片上层文件夹。')
    root = raw.expanduser().resolve()
    operation = context.parameters.get('operation', '分析并去重')
    if operation == '回滚最近一次':
        return _rollback_batch(root, context)
    if operation not in {'分析并去重', '仅分析（不移动文件）'}:
        raise ValueError('请选择有效的操作方式。')
    if context.parameters.get('dedup_mode') == '完整工序循环去重':
        raise ValueError('完整工序循环去重需要为各工位单独指定参考轮，请关闭“包含子文件夹”后逐目录处理。')
    folders = _folders(root, context)
    if not folders:
        raise ValueError('该目录及其子目录中没有支持的图片（已跳过备份目录和符号链接）。')
    from .image_process_dedup import PROCESS_MODE, load_regions, region_for
    if context.parameters.get('dedup_mode') == PROCESS_MODE:
        config = load_regions(context.parameters.get('process_regions'))
        if Path(config['root']) != root:
            raise ValueError('批量根目录已改变，请重新读取工位列表并框选区域。')
        for folder in folders:
            region_for(config, folder)  # Require every workstation BEFORE moving anything.
    preview = operation == '仅分析（不移动文件）'
    batch = {'version': 1, 'sourceFolder': str(root), 'preview': preview, 'status': 'running',
             'createdAt': datetime.now().isoformat(timespec='seconds'), 'folderCount': len(folders),
             'folderResults': []}
    batch_dir = _backup_root(root) / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}'
    batch_dir.mkdir(parents=True, exist_ok=False)
    report_path = batch_dir / 'batch.json'
    _write(report_path, batch)  # Ensure the index can be saved before any destructive work.
    context.report(f'找到 {len(folders)} 个图片目录，将分别处理，不跨目录比较。批量清单：{report_path}')
    try:
        for index, folder in enumerate(folders, 1):
            context.check_cancelled()
            relative = str(folder.relative_to(root))
            prefix = f'[{index}/{len(folders)} {relative}]'
            context.report(f'{prefix} 开始分析')
            child = replace(context, paths={**context.paths, 'images_folder': folder},
                            parameters={**context.parameters, 'recursive': False},
                            report=lambda message, p=prefix: context.report(f'{p} {message}'))
            try:
                result = _run_single_folder(child)
            except TaskCancelled:
                raise
            except Exception as exc:
                result = {'error': str(exc), 'message': f'失败：{exc}'}
                context.report(f'{prefix} 失败，跳过此目录并继续：{exc}')
            batch['folderResults'].append({'relativeFolder': relative, **result})
            _write(report_path, batch)
        batch['status'] = 'completed'
    except BaseException:
        batch['status'] = 'interrupted'
        _write(report_path, batch)
        context.report('批量处理已中断：当前目录未完成的移动已撤回；此前已完成目录的结果保留，可在同一上层目录回滚本批次。')
        raise
    finally:
        _write(report_path, batch)

    results = batch['folderResults']
    failed = sum('error' in item for item in results)
    if failed == len(results):
        raise ValueError(f'全部 {failed} 个图片目录处理失败，详见运行输出及批量清单：{report_path}')
    counts = {key: sum(item.get(key, 0) for item in results) for key in
              ('imageCount', 'keptCount', 'removedCount', 'unreadableCount')}
    proposed = sum(item.get('proposedRemovedCount', item.get('removedCount', 0)) for item in results)
    prefix = f'批量{"分析" if preview else "去重"}完成：成功 {len(results) - failed}/{len(results)} 个目录'
    detail = (f'建议保留 {counts["imageCount"] - proposed} 张，移除 {proposed} 张；尚未移动任何图片。' if preview else
              f'保留 {counts["keptCount"]} 张，移出 {counts["removedCount"]} 张，可回滚。')
    message = f'{prefix}；{detail}' + (f'另有 {failed} 个目录失败，请查看输出。' if failed else '')
    context.report(message)
    context.report(f'各目录统计及详细报告：{report_path}')
    return {'message': message, **counts, 'proposedRemovedCount': proposed, 'preview': preview,
            'folderCount': len(folders), 'successfulFolderCount': len(results) - failed,
            'failedFolderCount': failed, 'folderResults': results, 'batchReport': str(report_path),
            'outputFiles': [str(report_path)] + [p for item in results for p in item.get('outputFiles', [])],
            'outputFolders': [str(root)]}
