"""Stage lifecycle and measured suboperation progress, with an optional host sink."""
import logging
import time
from functools import wraps

log = logging.getLogger(__name__)
reporter = None


def emit(phase, message, completed=None, total=None):
    log.debug('%s: %s completed=%s total=%s', phase, message, completed, total)
    if reporter:
        reporter(phase, message, completed, total)


def stage(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        name, started = method.__name__, time.monotonic()
        log.info('Stage starting run=%s stage=%s', self.resolved_cfg.crucible_run_id, name)
        emit(name, f'Starting {name}')
        try:
            result = method(self, *args, **kwargs)
        except BaseException:
            log.exception('Stage failed stage=%s elapsed_seconds=%.3f', name, time.monotonic()-started)
            emit(name, f'Failed {name} after {time.monotonic()-started:.1f}s')
            raise
        log.info('Stage completed stage=%s elapsed_seconds=%.3f', name, time.monotonic()-started)
        emit(name, f'Completed {name} in {time.monotonic()-started:.1f}s')
        return result
    return wrapped
