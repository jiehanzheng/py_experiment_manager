#!/usr/bin/env python2

from __future__ import print_function, division
import atexit
import sys
import os
from multiprocessing import Pool, cpu_count
import subprocess
from collections import namedtuple, defaultdict, Counter


Task = namedtuple('Task', ['train_filename', 'model_filename', 'prediction_filename', 'svm_params'])
SVMReport = namedtuple('SVMReport', ['prediction_filename', 'time_train', 'time_test'])
DEV_NULL = open(os.devnull, 'w')


def train_and_test(task):
  if os.path.exists(task.prediction_filename):
    return task.prediction_filename

  os.nice(19)
  print("Training", task.model_filename, "with params", task.svm_params + "...", file=sys.stderr)
  svm_learn_process = subprocess.Popen(['/bin/bash', '-c', ' '.join(["svm_learn", task.svm_params, task.train_filename, task.model_filename])], stdout=DEV_NULL, stderr=DEV_NULL)
  atexit.register(svm_learn_process.terminate)
  svm_learn_process.wait()
  print("Testing using", task.model_filename + "...", file=sys.stderr)
  svm_classify_process = subprocess.Popen(['/bin/bash', '-c', ' '.join(["svm_classify", 'test', task.model_filename, task.prediction_filename])], stdout=DEV_NULL, stderr=DEV_NULL)
  atexit.register(svm_classify_process.terminate)
  svm_classify_process.wait()

  return task.prediction_filename


def count_for_f1(tp, fp, fn, tn, prediction_file, test_file, classes):
  for prediction_line, test_line in zip(prediction_file, test_file):
    actual_class = int(test_line.split(' ')[0])
    prediction_class = int(prediction_line.strip())

    # calculate tp, fp, fn, tn for each class
    for eval_class in classes:
      if actual_class == prediction_class:    # true
        if actual_class == eval_class:          # positive
          tp[eval_class] = tp[eval_class] + 1
        else:                                   # negative
          tn[eval_class] = tn[eval_class] + 1
      else:                                   # false
        if actual_class == eval_class:          # positive
          fp[eval_class] = fp[eval_class] + 1
        else:                                   # negative
          fn[eval_class] = fn[eval_class] + 1


if __name__ == "__main__":
  num_threads = cpu_count()

  if os.path.exists('svm_params'):
    with open('svm_params') as svm_params_file:
      svm_params = svm_params_file.readline()
  else:
    svm_params = ""    

  # find number of classes
  classes = set()
  with open('train') as train_file:
    for train_line in train_file:
      actual_class = int(train_line.split(' ')[0])
      classes.add(actual_class)

  num_classes = len(classes)

  # create one-vs-all files
  # TODO: implement pairwise
  for class_id in range(1, num_classes + 1):
    print("Producing", class_id, "vs all file...", file=sys.stderr)

    with open('train') as train_file:
      with open('train.class_' + str(class_id), 'w') as train_class_file:
        for train_line in train_file:
          line_components = train_line.split(' ')
          klass = int(line_components[0])

          if klass == class_id:
            klass = 1
          else:
            klass = -1

          line_components[0] = str(klass)
          train_class_file.write(' '.join(line_components))

  # delete train_file because it's not useful anymore
  os.remove(train_file.name)

  # find all train files and train N models with them
  queue = []
  for filename in os.listdir('.'):
    if filename.startswith('train') and len(filename) > len("train"):
      model_filename = filename.replace('train', 'model', 1)
      prediction_filename = filename.replace('train', 'prediction', 1)

      # find -j param
      num_pos = 0
      num_neg = 0
      with open(filename) as train_file:
        for train_line in train_file:
          klass = int(train_line.split(' ')[0])

          if klass == 1:
            num_pos = num_pos + 1
          else:
            num_neg = num_neg + 1

      this_svm_params = svm_params
      # this_svm_params = this_svm_params + ' -j ' + str(num_neg/num_pos)

      queue.append(Task(filename, model_filename, prediction_filename, this_svm_params))

  pool = Pool(processes=num_threads)
  prediction_filenames = pool.map(train_and_test, queue)

  # generate final prediction
  prediction_files = dict()
  for prediction_filename in prediction_filenames:
    class_id = int(prediction_filename[len('prediction.class_'):])
    prediction_files[class_id] = open(prediction_filename)

  print(prediction_files, file=sys.stderr)

  with open('prediction', 'w') as final_prediction_file:
    while True:
      class_lines = {class_id: prediction_file.readline().strip() for class_id, prediction_file in prediction_files.items()}

      try:
        max_key = max(class_lines.iterkeys(), key=lambda k: float(class_lines[k]))
      except ValueError:  # happens when float('') -> we reached EOF
        break
      
      max_score = float(class_lines[max_key])

      # TODO: add a flag
      # if max_score <= 0:
      #   best_class = -1
      # else:
      #   best_class = max_key
      best_class = max_key

      final_prediction_file.write(str(best_class) + '\n')

  for prediction_file in prediction_files.itervalues():
    prediction_file.close()

  # calculate f-measure and stuff
  tp = Counter()
  fp = Counter()
  fn = Counter()
  tn = Counter()
  with open('prediction') as prediction_file, open('test') as test_file:
    count_for_f1(tp=tp, fp=fp, fn=fn, tn=tn, prediction_file=prediction_file, test_file=test_file, classes=classes)

  # calculate f-measure for each class
  report = {}
  for class_id in classes:
    precision = tp[class_id]/(tp[class_id]+fp[class_id])
    recall = tp[class_id]/(tp[class_id]+fn[class_id])
    try:
      report[class_id] = {'f1': 2*(precision*recall)/(precision+recall), 'precision': precision, 'recall': recall}
    except ZeroDivisionError:
      report[class_id] = {'f1': "ZeroDivisionError", 'precision': precision, 'recall': recall}

  with open('json_result') as json_result_file:
    json_result.write(report)
