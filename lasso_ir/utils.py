import logging
from lasso_ir.constants import LOG_FILE, FILE_LOG_LEVEL, STDERR_LOG_LEVEL


class colors:
    RESET = "\x1b[39m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    CYAN = "\x1b[36m"


class Colorize(logging.Filter):
    def filter(self, record):
        color = colors.CYAN
        if record.levelno == logging.DEBUG:
            color = colors.GREEN
        elif record.levelno == logging.WARN:
            color = colors.YELLOW
        elif record.levelno >= logging.ERROR:
            color = colors.RED
        record.msg = "{}{}{}".format(color, record.msg, colors.RESET)
        record.levelname = "{}[{}]{}".format(color, record.levelname, colors.RESET)
        return True


LOGGER = logging.getLogger("{{cookiecutter.project_name}}")
LOGGER.setLevel(min(STDERR_LOG_LEVEL, FILE_LOG_LEVEL))

# log to file
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(FILE_LOG_LEVEL)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m-%d-%Y %H:%M%S"))

# log to stderr
stream_handler = logging.StreamHandler()
stream_handler.setLevel(STDERR_LOG_LEVEL)
stream_handler.addFilter(Colorize())
stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

LOGGER.addHandler(file_handler)
LOGGER.addHandler(stream_handler)


def warn(msg):
    LOGGER.log(logging.WARN, msg)


def info(msg):
    LOGGER.log(logging.INFO, msg)


def error(msg):
    LOGGER.log(logging.ERROR, msg)


def debug(msg):
    LOGGER.log(logging.DEBUG, msg)


# clean up
del Colorize
del file_handler
del stream_handler
