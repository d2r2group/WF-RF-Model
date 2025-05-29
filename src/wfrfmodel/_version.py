from importlib.metadata import version, PackageNotFoundError

version_fallback = "v0.0.0"

try:
    __version__ = version("wfrfmodel")
except PackageNotFoundError:
    # package is not installed
    __version__ = version_fallback
