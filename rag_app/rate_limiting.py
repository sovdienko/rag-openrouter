"""Rate limiting and retry utilities for API calls"""

import time
from functools import wraps
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded"""
    pass


class APIError(Exception):
    """Exception raised for general API errors"""
    pass


def rate_limit_retry(
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    backoff_factor: float = 2.0,
    retry_on: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff for rate limiting

    Implements exponential backoff strategy to handle API rate limits
    and temporary failures gracefully.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_wait: Initial wait time in seconds (default: 1.0)
        max_wait: Maximum wait time in seconds (default: 60.0)
        backoff_factor: Multiplier for wait time (default: 2.0)
        retry_on: Tuple of exception types to retry on (default: all exceptions)

    Returns:
        Decorated function with retry logic

    Examples:
        @rate_limit_retry(max_retries=3, initial_wait=1.0)
        def api_call():
            return client.embeddings.create(...)

        @rate_limit_retry(max_retries=5, backoff_factor=3.0)
        def critical_api_call():
            return client.chat.completions.create(...)

    Behavior:
        - First retry: wait initial_wait seconds (e.g., 1s)
        - Second retry: wait initial_wait * backoff_factor (e.g., 2s)
        - Third retry: wait initial_wait * backoff_factor^2 (e.g., 4s)
        - Capped at max_wait to prevent excessive delays
        - Detects rate limit errors from response messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            wait_time = initial_wait
            last_exception = None

            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded after {attempt} retries")
                    return result

                except retry_on as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Check if it's a rate limit error
                    is_rate_limit = any(
                        keyword in error_msg
                        for keyword in ["rate_limit", "rate limit", "429", "too many requests"]
                    )

                    # Check if it's a temporary error worth retrying
                    is_temporary = any(
                        keyword in error_msg
                        for keyword in ["timeout", "connection", "503", "502", "500"]
                    )

                    if attempt < max_retries - 1 and (is_rate_limit or is_temporary):
                        error_type = "Rate limit" if is_rate_limit else "Temporary error"
                        logger.warning(
                            f"{error_type} in {func.__name__} (attempt {attempt + 1}/{max_retries}), "
                            f"waiting {wait_time:.1f}s before retry..."
                        )
                        time.sleep(wait_time)

                        # Exponential backoff with cap
                        wait_time = min(wait_time * backoff_factor, max_wait)
                    else:
                        # Last attempt or non-retryable error
                        logger.error(f"{func.__name__} failed after {attempt + 1} attempts: {e}")
                        raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            return None

        return wrapper
    return decorator


class RateLimiter:
    """
    Token bucket rate limiter for API calls

    Implements token bucket algorithm to prevent exceeding API rate limits.
    Useful for controlling request rate before making API calls.

    Args:
        requests_per_minute: Maximum requests allowed per minute
        requests_per_second: Maximum requests allowed per second (optional)

    Examples:
        limiter = RateLimiter(requests_per_minute=60)

        for text in documents:
            limiter.wait_if_needed()
            embedding = service.embed_text(text)

        # Or with context manager
        with limiter:
            embedding = service.embed_text(text)
    """

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        requests_per_second: Optional[int] = None
    ):
        self.rpm = requests_per_minute
        self.rps = requests_per_second

        # Calculate minimum time between requests
        self.min_interval = 0.0
        if requests_per_second:
            self.min_interval = max(self.min_interval, 1.0 / requests_per_second)
        if requests_per_minute:
            self.min_interval = max(self.min_interval, 60.0 / requests_per_minute)

        self.last_request_time = 0.0
        self.request_count = 0

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        if self.min_interval == 0:
            return

        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

    def __enter__(self):
        """Context manager entry - wait if needed"""
        self.wait_if_needed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        return False

    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        return {
            "total_requests": self.request_count,
            "requests_per_minute": self.rpm,
            "requests_per_second": self.rps,
            "min_interval": self.min_interval,
            "last_request": self.last_request_time
        }


def adaptive_batch_size(
    total_items: int,
    initial_batch_size: int = 100,
    max_batch_size: int = 200,
    min_batch_size: int = 10
) -> list:
    """
    Generate adaptive batch sizes for processing

    Starts with larger batches and can reduce size if errors occur.
    Useful for finding optimal batch size dynamically.

    Args:
        total_items: Total number of items to process
        initial_batch_size: Starting batch size
        max_batch_size: Maximum allowed batch size
        min_batch_size: Minimum allowed batch size

    Returns:
        List of batch sizes for each iteration

    Examples:
        batch_sizes = adaptive_batch_size(1000, initial_batch_size=100)
        # Returns: [100, 100, 100, ...] for 10 batches

        # If error occurs, reduce batch size
        batch_sizes = adaptive_batch_size(1000, initial_batch_size=50)
    """
    batch_sizes = []
    remaining = total_items
    current_size = initial_batch_size

    while remaining > 0:
        # Clamp batch size to limits
        batch_size = max(min_batch_size, min(current_size, max_batch_size))
        batch_size = min(batch_size, remaining)

        batch_sizes.append(batch_size)
        remaining -= batch_size

    return batch_sizes


def retry_with_reduced_batch(
    func: Callable,
    items: list,
    initial_batch_size: int = 100,
    min_batch_size: int = 10,
    max_retries: int = 3
) -> list:
    """
    Retry function with progressively smaller batch sizes

    If processing fails with large batch, automatically retry with smaller batches.
    Useful for handling payload size limits.

    Args:
        func: Function to call with batch of items
        items: List of items to process
        initial_batch_size: Starting batch size
        min_batch_size: Minimum batch size to try
        max_retries: Maximum retry attempts per batch

    Returns:
        List of results from processing all items

    Examples:
        def process_batch(batch):
            return service.embed_texts(batch)

        results = retry_with_reduced_batch(
            func=process_batch,
            items=documents,
            initial_batch_size=100,
            min_batch_size=10
        )
    """
    results = []
    batch_size = initial_batch_size
    i = 0

    while i < len(items):
        batch = items[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                batch_results = func(batch)
                results.extend(batch_results)
                i += len(batch)
                break

            except Exception as e:
                error_msg = str(e).lower()

                # Check if error is related to payload size
                is_size_error = any(
                    keyword in error_msg
                    for keyword in ["payload", "too large", "size", "length"]
                )

                if is_size_error and batch_size > min_batch_size:
                    # Reduce batch size and retry
                    batch_size = max(min_batch_size, batch_size // 2)
                    logger.warning(
                        f"Payload too large, reducing batch size to {batch_size} and retrying..."
                    )
                elif attempt < max_retries - 1:
                    # Regular retry with backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"Error processing batch, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Max retries exceeded
                    logger.error(f"Failed to process batch after {max_retries} attempts: {e}")
                    raise

    return results
