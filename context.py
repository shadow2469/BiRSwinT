# --------------------------------------------------------
# Copyright (c) 2022 BiRSwinT Authors.
# Licensed under The MIT License.
# --------------------------------------------------------

class Context:
    # A Class provides configuration for training.

    def __init__(self):
        self.current_round = 0
        self.latest_round_result = None


ctx = Context()
