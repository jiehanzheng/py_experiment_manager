from __future__ import print_function, division
import atexit
import os
import sys
import re
from server_adapters.localhost import LocalRunner
from server_adapters.ssh import SSHRunner
from collections import namedtuple, defaultdict, Counter
from multiprocessing import JoinableQueue, Process, Manager, log_to_stderr
import logging
from prettytable import PrettyTable


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


def write_results_file(results):
  with open('results', 'w') as results_file:
    for job in results.keys():
      results_file.write(repr(job) + '\n')
      results_file.write("="*len(repr(job)) + '\n\n')
      results_file.write(str(results[job]) + '\n\n')


def print_table(results):
  # build a list of all svm_params
  # each being one table
  results_tables = defaultdict(Counter)

  # sum f1, precision, recall
  for job in results.keys():
    this_results_table = results_tables[job.svm_params]

    if 'classes' not in this_results_table:
      this_results_table['classes'] = defaultdict(Counter)

    this_results_table['num_exps'] += 1

    if not ('status' in results[job] and results[job]['status'] == 'waiting'):
      this_results_table['num_completed_exps'] += 1

      # sum f1, precision and recall for each class
      for class_id, stats in results[job].items():
        for stats_type, stats_value in stats.items():
          this_results_table['classes'][class_id][stats_type] += stats_value

  # divide by # of completed to get average
  for experiment_name in results_tables.keys():
    experiment_result = results_tables[experiment_name]
    for class_id, class_stats in experiment_result['classes'].items():
      for stats_type, stats_value in class_stats.items():
        experiment_result['classes'][class_id][stats_type] /= experiment_result['num_completed_exps']

  # TODO: sorting by class
  # TODO: stdev

  # build tables
  for table_name, result_table in results_tables.items():
    pretty_table = PrettyTable(["Class", "F1", "Precision", "Recall"])

    for class_id, stats in result_table['classes'].items():
      pretty_table.add_row([class_id, stats['f1'], stats['precision'], stats['recall']])

    print(table_name)
    print("="*len(table_name), end='\n\n')
    print(pretty_table, end='\n\n')


if __name__ == "__main__":
  os.chdir(sys.argv[1])

  # list experiment folder names
  experiment_folders = []
  for filename in os.listdir('.'):
    if os.path.isdir(filename) and get_fold_numbers(filename):
      print("Using fold combination:", filename, get_fold_numbers(filename), file=sys.stderr)
      experiment_folders.append(filename)
  

  logger = log_to_stderr()
  logger.setLevel(logging.DEBUG)

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
        servers_list.append(SSHRunner(username, hostname, directory, logger=logger))

  # read svm params list
  svm_params_list = []
  if os.path.exists('svm_params'):
    with open('svm_params') as svm_params_file:
      svm_params_list = [line.strip() for line in svm_params_file.readlines()]

  if len(svm_params_list) == 0:
    svm_params_list.append("")


  manager = Manager()
  results = manager.dict()

  # enqueue jobs
  job_queue = JoinableQueue()
  for svm_params in svm_params_list:
    for experiment_folder in experiment_folders:
      job = Job(directory=experiment_folder, svm_params=svm_params)
      job_queue.put(job)
      results[job] = {'status': 'waiting'}

  # ask servers to grab jobs
  for server in servers_list:
    t = Process(target=server.grab_jobs, args=(job_queue, results))
    t.daemon = True
    t.start()

  job_queue.join()

  write_results_file(results)
  print_table(results)
