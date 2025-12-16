# Version is managed by hatch-vcs from git tags
try:
    from importlib.metadata import version

    __version__ = version("aimnet")
except Exception:
    __version__ = "0.0.0+unknown"
