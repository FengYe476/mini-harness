import time
import random

from openai import APIConnectionError, InternalServerError, RateLimitError
from typing import Callable
from httpx import RemoteProtocolError

from mini_harness.config import CONFIG

def retry_call(request_agent: Callable, cfg = CONFIG):
    for attempt in range(max(cfg.max_retry, cfg.rate_retry)):
        try:
            return request_agent()
        except (APIConnectionError, InternalServerError, RateLimitError, RemoteProtocolError) as e:
            limited = isinstance(e, RateLimitError)
            allow = cfg.rate_retry if limited else cfg.max_retry
            if attempt >= allow - 1:
                raise
            base = cfg.rate_base if limited else cfg.retry_base
            wait = base * (2 ** attempt) + random.uniform(0, 1)
            if limited:
                wait = min(wait, cfg.rate_cap)
            print(f'[retry]: the api connection lost: {type(e).__name__}, waiting for retry -- {attempt +1 }/{cfg.max_retry} -- {wait:.1f}s')
            time.sleep(wait)
    raise ValueError(f'[invalid retries]: the max retry is invalid: {cfg.max_retry}')