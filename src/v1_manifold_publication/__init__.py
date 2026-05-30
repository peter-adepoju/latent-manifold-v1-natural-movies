"""Publication-upgrade tools for the V1 natural movie manifold project."""
from .readiness import build_readiness_report, format_readiness_markdown

__all__ = [
    "PublicationPaths",
    "ensure_publication_dirs",
    "build_readiness_report",
    "format_readiness_markdown",
]


def __getattr__(name):
    if name in {"PublicationPaths", "ensure_publication_dirs"}:
        from .paths import PublicationPaths, ensure_publication_dirs

        return {
            "PublicationPaths": PublicationPaths,
            "ensure_publication_dirs": ensure_publication_dirs,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
