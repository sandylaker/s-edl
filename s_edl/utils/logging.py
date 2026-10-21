"""Global logger utilities with colored output and distributed support.

Based on PyTorch Ignite's setup_logger:
https://docs.pytorch.org/ignite/_modules/ignite/utils.html#setup_logger
"""
import logging
import os
import sys
from typing import TextIO


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for terminal output.

    Colors:
        - DEBUG: Cyan
        - INFO: Green
        - WARNING: Yellow
        - ERROR: Red
        - CRITICAL: Bold Red
    """

    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[1;31m',  # Bold Red
    }
    RESET = '\033[0m'

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_color: bool = True,
    ) -> None:
        """Initialize the formatter.

        Args:
            fmt: Log message format string.
            datefmt: Date format string.
            use_color: Whether to use colored output.
        """
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with optional color.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string with color codes if enabled.
        """
        # Make a copy to avoid modifying the original record
        record = logging.makeLogRecord(record.__dict__)

        if self.use_color and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            record.levelname = f'{color}{record.levelname}{self.RESET}'

        return super().format(record)


def setup_logger(
    name: str | None = 's_edl',
    level: int = logging.INFO,
    stream: TextIO | None = None,
    fmt: str = '%(asctime)s %(name)s %(levelname)s: %(message)s',
    datefmt: str | None = None,
    filepath: str | None = None,
    distributed_rank: int | None = None,
    reset: bool = False,
    encoding: str = 'utf-8',
    use_color: bool = True,
) -> logging.Logger:
    """Setup a logger with colored output and distributed environment support.

    In distributed training, only rank 0 will log messages by default.
    Other ranks will have a NullHandler to suppress output.

    Args:
        name: Logger name. If None, returns the root logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        stream: Output stream. Defaults to sys.stderr.
        fmt: Log message format string.
        datefmt: Date format string for timestamps.
        filepath: Optional file path to also write logs to.
        distributed_rank: Process rank in distributed training.
            If None, will try to detect from environment variables or
            torch.distributed (if initialized).
            Set to 0 to force logging on the current process.
        reset: If True, removes existing handlers and reconfigures the logger.
        encoding: File encoding when filepath is specified.
        use_color: Whether to use colored output in terminal.

    Returns:
        Configured logger instance.

    Example:
        >>> from s_edl.utils.logging import setup_logger
        >>> logger = setup_logger('my_module', level=logging.DEBUG)
        >>> logger.info('Training started')
        >>> logger.warning('Learning rate is high')
    """
    # Check if logger already exists
    existing = name is None or name in logging.root.manager.loggerDict

    # Get or create logger
    logger = logging.getLogger(name)

    # Auto-detect distributed rank
    if distributed_rank is None:
        distributed_rank = get_dist_rank()

    # Remove previous handlers for non-zero ranks or when resetting
    if distributed_rank > 0 or reset:
        if logger.hasHandlers():
            for handler in list(logger.handlers):
                logger.removeHandler(handler)

    # Non-zero ranks get a null handler (suppress output)
    if distributed_rank > 0:
        logger.addHandler(logging.NullHandler())
        return logger

    # Keep existing configuration if not resetting
    if existing and not reset:
        return logger

    # Configure rank 0 logger
    logger.setLevel(level)

    # Determine if we should use color (only for TTY streams)
    if stream is None:
        stream = sys.stderr
    should_use_color = use_color and _is_tty(stream)

    # Create and add stream handler
    stream_handler = logging.StreamHandler(stream=stream)
    stream_handler.setLevel(level)
    stream_formatter = ColoredFormatter(fmt, datefmt, use_color=should_use_color)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # Add file handler if filepath specified (no colors for files)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, encoding=encoding)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt, datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    if name is not None:
        logger.propagate = False

    return logger


def get_dist_rank() -> int:
    """Get the distributed rank of the current process.

    Detection priority:
    1. Environment variables (RANK, LOCAL_RANK, etc.) - works before dist init
    2. torch.distributed.get_rank() - if already initialized

    Returns:
        The process rank, or 0 if not in a distributed environment.
    """
    # First, try environment variables (most reliable, works before dist init)
    # These are set by torchrun, accelerate launch, etc.
    rank_env_vars = ['RANK', 'LOCAL_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK']

    for var in rank_env_vars:
        rank = os.environ.get(var)
        if rank is not None:
            try:
                return int(rank)
            except ValueError:
                continue

    # Fallback: try torch.distributed if initialized
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except ImportError:
        pass

    return 0


def _is_tty(stream: TextIO) -> bool:
    """Check if a stream is a TTY (terminal).

    Args:
        stream: The stream to check.

    Returns:
        True if the stream is a TTY, False otherwise.
    """
    return hasattr(stream, 'isatty') and stream.isatty()
