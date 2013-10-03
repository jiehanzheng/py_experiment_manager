from __future__ import print_function
import os
import sys
import re
from server_adapters.localhost import LocalRunner
from server_adapters.ssh import SSHRunner
from collections import namedtuple
from multiprocessing import Queue, Process


Job = namedtuple('Job', ['directory', 'svm_params'])


folder_name_pattern = re.compile(r'_model(?P<fold_numbers>(?:_\d+)+)$')
train_filename_pattern = re.compile(r'^train\.class_(?P<class>\d+)$')


def get_class_number_string(train_filename):
  result = train_filename_pattern.match(train_filename)
  if result:
    return result.group('class')
  else:
    return False


def get_fold_numbers(folder_name):
  result = folder_name_pattern.search(folder_name)
  if result:
    fold_number_string = result.group('fold_numbers')
    fold_number_string = fold_number_string[1:]
    return [int(fold_number) for fold_number in fold_number_string.split('_')]
  else:
    return False


if __name__ == "__main__":
  os.chdir(sys.argv[1])

  # list experiment folder names
  experiment_folders = []
  for filename in os.listdir('.'):
    if os.path.isdir(filename) and get_fold_numbers(filename):
      print("Using fold combination:", filename, get_fold_numbers(filename), file=sys.stderr)
      experiment_folders.append(filename)
  
  # read available server lists
  servers_list = []
  with open("servers_list") as servers_file:
    for server_line in servers_file:
      if server_line.startswith('#'):
        continue

      hostname_and_directory = server_line.strip()
      hostname, directory = hostname_and_directory.split(':')

      if hostname == 'localhost':
        servers_list.append(LocalRunner(directory))
      else:
        username, hostname = hostname.split('@')
        servers_list.append(SSHRunner(username, hostname, directory))

  # read svm params list
  svm_params_list = []
  if os.path.exists('svm_params'):
    with open('svm_params') as svm_params_file:
      svm_params_list = [line.strip() for line in svm_params_file.readlines()]

  if len(svm_params_list) == 0:
    svm_params_list.append("")

  # enqueue jobs
  job_queue = Queue()
  for svm_params in svm_params_list:
    for experiment_folder in experiment_folders:
      # create svm_params file
      if len(svm_params) > 0:
        with open(os.path.join(experiment_folder, 'svm_params'), 'w') as svm_params_file:
          svm_params_file.write(svm_params)

      job_queue.put(Job(directory=experiment_folder, svm_params=svm_params))

  # ask servers to grab jobs
  results = dict()
  for server in servers_list:
    t = Process(target=server.grab_jobs(job_queue, results))
    t.daemon = True
    t.start()

  job_queue.join()

  # process results
  with open('results') as results_file:
    for job, result_json in results.iteritems():
      results_file.write(repr(job) + '\n')
      results_file.write(len(repr(job)) + '\n\n')
      results_file.write(result_json + '\n')
