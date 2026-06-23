"""统一日志配置。全项目用这个 logger，别到处 print。"""
import logging
import sys


def setup_logger(name: str = "doclens") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # 避免重复添加 handler
        return logger
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.stream.reconfigure(encoding="utf-8", errors="replace")  # Windows 中文
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()
