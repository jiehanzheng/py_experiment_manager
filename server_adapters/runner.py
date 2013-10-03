from abc import ABCMeta, abstractmethod

class Runner():
  __metaclass__ = ABCMeta
  
  @abstractmethod
  def _do_experiment(self, folder_name): pass

  def grab_jobs(self, queue):
    self._do_experiment(queue.get().directory)
    queue.task_done()
