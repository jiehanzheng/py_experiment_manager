from abc import ABCMeta, abstractmethod

class Runner():
  __metaclass__ = ABCMeta
  
  @abstractmethod
  def do_experiment(self, folder_name): pass

  @abstractmethod
  def _cleanup(self): pass

  def grab_jobs(self, queue, results):
    while True:
      directory = queue.get().directory
      result = self.do_experiment(directory)
      results.append(result)
      self._cleanup()
      queue.task_done()
