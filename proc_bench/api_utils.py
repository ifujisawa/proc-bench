from time import sleep
import google.api_core.exceptions

__EXCEPTIONS__ = (
    google.api_core.exceptions.ResourceExhausted,
)

NUM_RETRIES = 5
SLEEP_DURATION = 60


def sleep_and_retry(func):
    def wrapper(*args, **kwargs):
        for _ in range(NUM_RETRIES):
            try:
                return func(*args, **kwargs)
            except __EXCEPTIONS__ as e:
                print(f"Caught exception: {e}")
                for t in range(SLEEP_DURATION):
                    print(f"Retrying in {SLEEP_DURATION - t} seconds...", end="\r")
                    sleep(1)
                    
        raise Exception(f"Failed after {NUM_RETRIES} retries.")
                    
    return wrapper