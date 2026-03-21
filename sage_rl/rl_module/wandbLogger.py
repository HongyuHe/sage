import time
import os

import wandb
from acme.utils.loggers import base


def _print_wandb_run_links(run, *, project_name=None):
    if run is None:
        return

    project_url = ""
    run_url = ""
    project_url_getter = getattr(run, "project_url", None)
    if callable(project_url_getter):
        try:
            project_url = str(project_url_getter() or "")
        except Exception:
            project_url = ""
    run_url_getter = getattr(run, "get_url", None)
    if callable(run_url_getter):
        try:
            run_url = str(run_url_getter() or "")
        except Exception:
            run_url = ""
    if not run_url:
        try:
            run_url = str(getattr(run, "url", "") or "")
        except Exception:
            run_url = ""
    if not project_url and run_url and "/runs/" in run_url:
        project_url = str(run_url.split("/runs/", 1)[0])
    if not project_url and project_name:
        entity = str(getattr(run, "entity", "") or "")
        if entity:
            project_url = f"https://wandb.ai/{entity}/{project_name}"
    if project_url:
        print(f"wandb: ⭐️ View project at {project_url}")
    if run_url:
        print(f"wandb: 🚀 View run at {run_url}")


class wandbLogger(base.Logger):
    def __init__(self, job_type, config, group, time_delta=0.0, name=None):


        name = config['environment_name'] + "_" + config['wandb']['proj_name']
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")
        run = wandb.init(project=name, config=config, resume=False, group=group, job_type=job_type)
        _print_wandb_run_links(run, project_name=name)
        self._time = time.time()
        self._time_delta = time_delta
        if "Actor" not in job_type:
            self._counter = None
        else:
            self._counter = dict(Actor_steps=0)
    def write(self, data):
        now = time.time()
        if (now - self._time) > self._time_delta:
            if self._counter is not None:
                data.update(self._counter)
            wandb.log(data)
            self._time = now

    def actor_steps_counter_update(self, count):
        self._counter['Actor_steps'] = count
