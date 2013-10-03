from __future__ import print_function
from .runner import Runner
import sys
import os

class LocalRunner(Runner):

  def __init__(self, working_directory):
    raise NotImplementedError()

  def do_experiment(self, folder_name):
    raise NotImplementedError()

  def _cleanup():
    raise NotImplementedError()
