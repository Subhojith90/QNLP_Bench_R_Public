from __future__ import annotations

from typing import Any, Sequence


class OptionalQuantumDependencyError(ImportError):
    """Raised when an optional QNLP dependency such as lambeq is requested but unavailable."""


def check_lambeq_available() -> bool:
    try:
        import lambeq  # noqa: F401
        return True
    except Exception:
        return False


def require_lambeq() -> Any:
    try:
        import lambeq
        return lambeq
    except Exception as exc:
        raise OptionalQuantumDependencyError("lambeq is not installed. Install it explicitly for real DisCoCat/QNLP parsing experiments; the synthetic core does not require it.") from exc


def parse_sentences_lambeq(sentences: Sequence[str]) -> list[Any]:
    lambeq = require_lambeq()
    parser = lambeq.BobcatParser(verbose="suppress")
    return parser.sentences2diagrams(list(sentences))
