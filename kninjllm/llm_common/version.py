from importlib import metadata

try:
    __version__ = str(metadata.version("kninjllm-ai"))
except metadata.PackageNotFoundError:
    __version__ = "main"
