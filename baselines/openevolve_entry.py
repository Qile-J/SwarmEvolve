"""OpenEvolve's entry point (openevolve-run.py) with one fix: its SIGTERM handler no longer deadlocks. The run's budget ends with
the SIGTERM of `timeout`; the search itself is unchanged."""
from pathlib import Path
import sys

# import OpenEvolve exactly as its own entry point does (from its repository root): from baselines/, the clone's directory
# baselines/openevolve would shadow the package
sys.path[0] = str(Path(__file__).resolve().parent / 'openevolve')
import threading  # noqa: E402

from openevolve.process_parallel import ProcessParallelController, logger  # noqa: E402


def request_shutdown(self):
    # OpenEvolve's SIGTERM handler calls this, which sets a multiprocessing.Event; the evolution loop holds that event's lock
    # every 10 ms (is_set), and a signal arriving then deadlocks the process. Setting it from a thread lets the handler return
    # at once; the loop sees the event on its next check, as before.
    logger.info('Graceful shutdown requested...')
    threading.Thread(target=self.shutdown_event.set, daemon=True).start()


ProcessParallelController.request_shutdown = request_shutdown

if __name__ == '__main__':
    from openevolve.cli import main
    sys.exit(main())
