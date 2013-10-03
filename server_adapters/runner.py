from abc import ABCMeta, abstractmethod, abstractproperty

class Runner():
  __metaclass__ = ABCMeta

  @abstractproperty
  def logger(self): pass

  @abstractmethod
  def do_experiment(self, folder_name): pass

  @abstractmethod
  def _cleanup(self): pass

  def grab_jobs(self, queue, results):
    while True:
      job = queue.get()
      result = self.do_experiment(job.directory)
      self.logger.debug(str(job) + ': ' + str(result))
      results[job] = result
      queue.task_done()
