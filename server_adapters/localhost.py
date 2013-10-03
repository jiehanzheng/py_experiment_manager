from __future__ import print_function
from .runner import Runner
import sys
import os

class LocalRunner(Runner):

  def __init__(self, working_directory):
    self.working_directory = working_directory

    print("Preparing localhost...", file=sys.stderr)
    raise NotImplementedError()

  def _do_experiment(self, folder_name):
    raise NotImplementedError()
