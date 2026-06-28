"""统一日志配置。全项目用这个 logger，别到处 print。"""
import logging
import sys


def setup_logger(name: str = "doclens") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.stream.reconfigure(encoding="utf-8", errors="replace")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def get_logger(trace_id: str = "") -> logging.Logger:
    """获取带 trace_id 的 logger（多智能体链路追踪用）"""
    if trace_id:
        return logging.getLogger(f"doclens.{trace_id}")
    return logger
