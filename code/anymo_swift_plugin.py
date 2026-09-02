from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from transformers import TrainerCallback
    from swift.callbacks.mapping import callbacks_map
except ModuleNotFoundError:
    TrainerCallback = object
    callbacks_map = {}


class AnyMoRowFreezeCallback(TrainerCallback):
    def __init__(self, args=None, trainer=None):
        super().__init__()
        self.args = args
        self.trainer = trainer

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None or model is None or not hasattr(model, 'collect_anymo_diagnostics'):
            return control
        diagnostics = model.collect_anymo_diagnostics()
        logs.update(diagnostics)
        return control


callbacks_map['anymo_row_freeze'] = AnyMoRowFreezeCallback
