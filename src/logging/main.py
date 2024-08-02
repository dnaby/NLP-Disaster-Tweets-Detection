import structlog
import socket


class LoggerManager:
    def __init__(self, filename='main.py'):
        self.hostname = socket.gethostname()
        self.ip_address = socket.gethostbyname(self.hostname)
        self.configure_structlog(filename)

    def configure_structlog(self, filename):
        # Clear any existing context variables
        structlog.contextvars.clear_contextvars()
        
        # Bind context variables
        structlog.contextvars.bind_contextvars(
            ip_address=self.ip_address,
            hostname=self.hostname,
            filename=filename
        )
        
        self.log = structlog.get_logger(__name__)

    def info(self, message, **extra_vars):
        self.log.info(message, **extra_vars)
        
    def debug(self, message, **extra_vars):
        self.log.debug(message, **extra_vars)

    def warning(self, message, **extra_vars):
        self.log.warning(message, **extra_vars)

    def error(self, message, **extra_vars):
        self.log.error(message, **extra_vars)

    def critical(self, message, **extra_vars):
        self.log.critical(message, **extra_vars)
