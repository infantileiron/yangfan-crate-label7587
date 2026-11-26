import json
import logging
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.config.settings import ARTIFACTS
from src.models.database import cache
from src.services.domain import compute
from src.services.domain_rolling_store import RollingStore
from src.services.workflow import claim_run, finish_run, is_rolling, publish_due
from src.utils.exports import write_csv

ROLLING_STATE_DIR = ARTIFACTS / "rolling-state"


def calculate(job):
    key, claim, rows, options = job
    rolling = is_rolling(options)
    try:
        if rolling:
            locked_options = {name: value for name, value in options.items() if name != "mode"}
            report = RollingStore(ROLLING_STATE_DIR).run(rows, locked_options)
            json.dumps(report, allow_nan=False)
            artifact = write_csv(report['selected_per_window'],
                                 list(report['selected_per_window'][0]) if report['selected_per_window']
                                 else ['窗口'])
        else:
            report = compute(rows, options)
            report['columns'] = list(dict.fromkeys(key for row in report['rows'] for key in row))
            report['metric_order'] = list(report['metrics'])
            json.dumps(report, allow_nan=False)
            artifact = write_csv(report['rows'], report['columns'])
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=ARTIFACTS, delete=False) as temporary:
            temporary.write(artifact)
        Path(temporary.name).replace(ARTIFACTS / f'run-{key}-{claim.hex()}.csv')
    except ValueError as error:
        finish_run(key, claim, None, str(error))
    except Exception:
        logging.exception('Analysis %s failed', key)
        finish_run(key, claim, None, '计算失败，请核对数据并重新提交。')
    else:
        finish_run(key, claim, report, None)


def work():
    while True:
        try:
            job = claim_run()
            if job:
                calculate(job)
            publish_due()
            cache.set('worker:heartbeat', str(time.time()), ex=25)
        except Exception:
            logging.exception('Worker iteration failed')
        time.sleep(0.8)


if __name__ == '__main__':
    work()
