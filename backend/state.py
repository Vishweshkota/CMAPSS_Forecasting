from collections import deque
import numpy as np


# Maintains rolling feature history for each engine during streaming inference.
class EngineStateManager:
    def __init__(self, window_size):
        self.window_size = window_size
        self.engine_buffers = {}

    # Create a unique key so engines from different dataset tags stay separate.
    def _make_key(self, tag, engine_id):
        return f"{tag}::{engine_id}"

    # Add one processed row to the correct engine buffer.
    # The buffer automatically keeps only the latest `window_size` rows.
    def add_processed_row(self, tag, engine_id, processed_row):
        key = self._make_key(tag, engine_id)

        if key not in self.engine_buffers:
            self.engine_buffers[key] = deque(maxlen=self.window_size)

        row_array = processed_row.iloc[0].to_numpy(dtype=np.float32)
        self.engine_buffers[key].append(row_array)

        return len(self.engine_buffers[key])

    # Check whether enough rows are available to form a prediction window.
    def is_window_ready(self, tag, engine_id):
        key = self._make_key(tag, engine_id)

        if key not in self.engine_buffers:
            return False

        return len(self.engine_buffers[key]) == self.window_size

    # Return the stacked rolling window for one engine.
    def get_window(self, tag, engine_id):
        key = self._make_key(tag, engine_id)

        if not self.is_window_ready(tag, engine_id):
            raise ValueError(f"Window not ready for {key}")

        return np.stack(self.engine_buffers[key], axis=0)

    # Return the current number of stored rows for one engine.
    def get_buffer_length(self, tag, engine_id):
        key = self._make_key(tag, engine_id)

        if key not in self.engine_buffers:
            return 0

        return len(self.engine_buffers[key])

    # Remove one engine's buffered history.
    def reset_engine(self, tag, engine_id):
        key = self._make_key(tag, engine_id)

        if key in self.engine_buffers:
            del self.engine_buffers[key]

    # Clear all engine histories.
    def reset_all(self):
        self.engine_buffers.clear()
