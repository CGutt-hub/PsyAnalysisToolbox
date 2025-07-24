import logging

def setup_logging(config):
    log_level = config.get('DEFAULT', 'log_level', fallback='INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger()

def get_logger(name):
    return logging.getLogger(name) 