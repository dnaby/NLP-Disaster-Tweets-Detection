import structlog
import socket

class LoggerManager:
    """
    A class to manage logging operations with structured logging.
    """

    def __init__(self, filename: str = 'main.py'):
        """
        Initializes the LoggerManager with a filename for logging context.

        Parameters:
        filename (str): The filename to be used in logging context. Defaults to 'main.py'.
        """
        self.hostname = socket.gethostname()
        self.ip_address = socket.gethostbyname(self.hostname)
        self.configure_structlog(filename)

    def configure_structlog(self, filename: str):
        """
        Configures structured logging with context variables.

        Parameters:
        filename (str): The filename to be used in logging context.
        """
        # Clear any existing context variables
        structlog.contextvars.clear_contextvars()
        
        # Bind context variables
        structlog.contextvars.bind_contextvars(
            ip_address=self.ip_address,
            hostname=self.hostname,
            filename=filename
        )
        
        self.log = structlog.get_logger(__name__)

    def info(self, message: str, **extra_vars) -> None:
        """
        Logs an info message with optional extra variables.

        Parameters:
        message (str): The message to be logged.
        **extra_vars: Optional extra variables to be included in the log.
        """
        self.log.info(message, **extra_vars)
        
    def debug(self, message: str, **extra_vars) -> None:
        """
        Logs a debug message with optional extra variables.

        Parameters:
        message (str): The message to be logged.
        **extra_vars: Optional extra variables to be included in the log.
        """
        self.log.debug(message, **extra_vars)

    def warning(self, message: str, **extra_vars) -> None:
        """
        Logs a warning message with optional extra variables.

        Parameters:
        message (str): The message to be logged.
        **extra_vars: Optional extra variables to be included in the log.
        """
        self.log.warning(message, **extra_vars)

    def error(self, message: str, **extra_vars) -> None:
        """
        Logs an error message with optional extra variables.

        Parameters:
        message (str): The message to be logged.
        **extra_vars: Optional extra variables to be included in the log.
        """
        self.log.error(message, **extra_vars)

    def critical(self, message: str, **extra_vars) -> None:
        """
        Logs a critical message with optional extra variables.

        Parameters:
        message (str): The message to be logged.
        **extra_vars: Optional extra variables to be included in the log.
        """
        self.log.critical(message, **extra_vars)
