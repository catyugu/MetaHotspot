"""单例日志器实现。"""

import logging
import sys

class SimLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger = logging.getLogger("MetaHotspot")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            cls._instance.logger = logger
        return cls._instance

def get_logger() -> logging.Logger:
    return SimLogger().logger