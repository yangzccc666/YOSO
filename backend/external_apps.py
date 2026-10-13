"""Compatibility wrappers for launching configurable external apps."""

from .app_settings import settings_store


def launch_xanylabeling():
    return settings_store.launch("xanylabeling")


def launch_antigravity():
    return settings_store.launch("antigravity")
