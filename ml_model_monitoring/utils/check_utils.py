from functools import wraps

from .logger_utils import get_logger


def validate_params(param_types: dict):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(__name__)

            for param, p_type in param_types.items():
                if param in kwargs:
                    if not isinstance(kwargs[param], p_type):
                        logger.error(
                            f"Invalid type for {param}. Expected {p_type}, "
                            f"got {type(kwargs[param])}"
                        )
                        raise ValueError(
                            f"Invalid type for {param}. Expected {p_type}, "
                            f"got {type(kwargs[param])}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator
