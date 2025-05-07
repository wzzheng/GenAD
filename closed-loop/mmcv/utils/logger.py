import logging
from .logging import get_logger
from mmcv import __version__

def get_root_logger(log_file=None, log_level=logging.INFO, name = __version__):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger
