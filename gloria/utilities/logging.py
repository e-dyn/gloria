"""
Definition of Gloria logger and its configuration
"""

# Standard Library
import json
import locale
import logging
import multiprocessing
import os
import platform
import pprint
import shutil
import subprocess
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Union

# Third Party
from pydantic import BaseModel, ConfigDict, field_validator

# Gloria
from gloria.utilities.constants import _GLORIA_PATH, _RUN_TIMESTAMP
from gloria.utilities.types import LogLevel

### --- Logger settings --- ###
# The logging levels for stream and file logs


class LoggingConfig(BaseModel):
    model_config = ConfigDict(
        # Use validation also when fields of an existing model are assigned
        validate_assignment=True
    )

    stream_level: LogLevel = "INFO"
    file_level: LogLevel = "DEBUG"
    log_path: Path = _GLORIA_PATH / "logfiles"
    write_logfile: bool = True

    @field_validator("log_path", mode="before")
    @classmethod
    def validate_log_path(cls, log_path: Union[Path, str]) -> Path:
        try:
            log_path = Path(log_path)
        except Exception as e:
            raise ValueError(
                f"Cannot convert log_path input {log_path} to a path."
            ) from e
        return log_path


log_config = LoggingConfig()


def error_with_traceback(func: Callable[[str], Any]) -> Callable[[str], Any]:
    """
    A decorator for the logger.error function, adding a traceback from the
    invocation point to the latest call.

    Parameters
    ----------
    func : Callable[str, Any]
        The logging.error function to be decorated

    Returns
    -------
    Callable[str, Any]
        The decorated error function

    """

    # Define the wrapper for the error function
    def wrapper(
        msg: str, *args: tuple[Any, ...], **kwargs: dict[Any, Any]
    ) -> None:
        # Get the path of the main script
        # Third Party
        import __main__

        main_script = str(Path(__main__.__file__)).lower()

        # Walk through the entire traceback and extract the filepaths
        traceback_files = [
            frame.f_code.co_filename.lower()
            for frame, _ in traceback.walk_stack(None)
        ]
        traceback_files = traceback_files[::-1]

        # Find the index of the invocation point, ie. the first time the main
        # script was executed
        main_script_index = min(
            [
                idx
                for idx, file in enumerate(traceback_files)
                if main_script == file
            ]
        )

        # Use the index to filter the traceback and append it to the original
        # error message. The slice runs only until -1 to remove the call of
        # the wrapper itself
        msg = f"{msg}\n" + "".join(
            traceback.format_stack()[main_script_index:-1]
        )
        return func(msg, *args, **kwargs)

    return wrapper


def collect_sysinfo() -> str:
    """
    Collect basic information about system and environment Gloria was run in.

    Returns
    -------
    str
        Formated string containing all collected information about the system

    """
    # Collect system infos
    info = {
        "pointer_size": "64bit" if sys.maxsize > 2**32 else "32bit",
        "binary_format": platform.architecture()[1],
        "machine": platform.machine(),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "cpu_count": multiprocessing.cpu_count(),
        "processor": platform.processor() or platform.machine(),  # Fallback
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": sys.version,
        "python_executable": sys.executable,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "locale": locale.getdefaultlocale(),
        "preferred_encoding": locale.getpreferredencoding(False),
        "timezone": time.tzname,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # List all installed packages along with versions
    try:
        pkgs = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pip",
                "list",
                "--format=json",
                "--disable-pip-version-check",
            ],
            text=True,
        )
        info["packages"] = json.loads(pkgs)
    except Exception:
        info["packages"] = "unavailable"

    # In case Gloria was executed from a git repo, identify the commit
    # Identify full git exectuable path
    git_exe = shutil.which("git")

    if git_exe:
        try:
            git_root = subprocess.check_output(
                [git_exe, "rev-parse", "--show-toplevel"], text=True
            ).strip()
            commit = subprocess.check_output(
                [git_exe, "-C", git_root, "rev-parse", "HEAD"], text=True
            ).strip()
            info["git_commit"] = commit
        except Exception:
            info["git_commit"] = "unavailable"
    else:
        info["git_commit"] = "unavailable"
    out = pprint.pformat(info, sort_dicts=False)
    return out


@lru_cache(maxsize=None)
def get_logger() -> logging.Logger:
    """
    Set up the gloria logger for sending log entries both to the stream and
    a log file.

    The logger has a static name except for a timestamp when the main script
    was executed. Hence, the logger will be unique for a single session across
    all gloria modules.

    Parameters
    ----------
    log_path : Path, optional
        The path the log file will be saved to.
        The default is _GLORIA_PATH / 'logfiles'.
    timestamp : pd.Timestampstr, optional
        A timestamp to integrate into the logger name. The default is
        _RUN_TIMESTAMP, which is the time he main script was executed.

    Returns
    -------
    logging.Logger
        The configured Logger

    """

    def stream_filter(record):
        """
        Keep only logs of level WARNING or below
        """
        return record.levelno <= logging.WARNING

    # Get the logger. Note that the timestamp is part of the logger name. If
    # timestamp uses the default _RUN_TIMESTAMP the logger will be unique
    # for a single session and all modules will use the same logger.
    logger = logging.getLogger(f"gloria_{_RUN_TIMESTAMP}")
    # Set the level of the root logger to debug so nothing gets lost
    logger.setLevel("DEBUG")

    # A common format for all log entries
    formatter = logging.Formatter(
        "{asctime} - gloria - {levelname} - {message}",
        style="{",
        datefmt="%H:%M:%S",
    )

    # Configure the stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_config.stream_level)
    # Don't show errors in the stream, as python will take care of it
    stream_handler.addFilter(stream_filter)
    # Set stream format
    stream_handler.setFormatter(formatter)
    # Add stream handler
    logger.addHandler(stream_handler)

    # Configue the file handler
    if log_config.write_logfile:
        # Prepare log-path and file
        log_dir = Path(log_config.log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{_RUN_TIMESTAMP}.log"
        Path(log_config.log_path).mkdir(parents=True, exist_ok=True)
        # Write system info to file
        with open(log_file, "w") as file:
            output = (
                "\n\n".join(
                    [
                        " System Info ".center(50, "-"),
                        collect_sysinfo(),
                        " Execution Log ".center(50, "-"),
                    ]
                )
                + "\n\n"
            )
            file.write(output)
        file_handler = logging.FileHandler(
            log_file,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(log_config.file_level)
        # Set file format
        file_handler.setFormatter(formatter)
        # Add file handler
        logger.addHandler(file_handler)

    # Decorate the loggers error method so it will include error tracebacks
    # in the log-file.
    # Note ruff doesn't like the setattr, hence the noqa. With direct
    # assignment mypy complains. No way to make everyone happy.
    setattr(logger, "error", error_with_traceback(logger.error))  # noqa: B010
    return logger
