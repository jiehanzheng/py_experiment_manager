#!/usr/bin/env python2

from __future__ import print_function, division
import atexit
import sys
import os
from multiprocessing import Pool, cpu_count
import subprocess
from collections import namedtuple, defaultdict, Counter
import json
import re


DEV_NULL = open(os.devnull, 'w')


def get_text_section(outputdata, section_name):
  section_content = ""
  num_consecutive_blank_lines = 0
  in_section = False

  for output_line in outputdata.splitlines():
    print(output_line)

    if output_line.strip() == "=== " + section_name + " ===":
      in_section = True
      continue

    if in_section and len(output_line.strip()) == 0:
      num_consecutive_blank_lines += 1

    if in_section and num_consecutive_blank_lines <= 2:
      section_content += output_line + '\n'

  return section_content


valid_class_report_component = re.compile(r'(\d|\.)+')

def is_class_report_line(report_line):
  if len(report_line.strip()) == 0:
    return False

  report_line_components = report_line.split()
  return all(valid_class_report_component.match(component) for component in report_line_components)


def _kill(process):
  print("Killing", str(process) + '...', file=sys.stderr)
  try:
    os.kill(process.pid, 9)
  except OSError:
    pass


if __name__ == "__main__":
  num_threads = cpu_count()
  svm_classifier = sys.argv[1]
  svm_params = sys.argv[2]

  os.nice(19)

  # TODO: change to weka commands
  print("Training", 'model.arff', "with params", svm_params + "...", file=sys.stderr)
  svm_learn_process = subprocess.Popen(['/bin/bash', '-c', ' '.join(["java", '-Xmx4g', svm_classifier, svm_params, '-t', 'train.arff', '-d', 'model'])], stdout=DEV_NULL, stderr=DEV_NULL)
  atexit.register(_kill, svm_learn_process)
  svm_learn_process.wait()

  # TODO: test and parse weka result tables
  print("Testing using", 'model' + "...", file=sys.stderr)
  svm_classify_process = subprocess.Popen(['/bin/bash', '-c', ' '.join(["java", '-Xmx4g', svm_classifier, '-i', '-l', 'model', '-T', 'test.arff'])], stdout=subprocess.PIPE, stderr=DEV_NULL)
  atexit.register(_kill, svm_classify_process)
  stdoutdata, stderrdata = svm_classify_process.communicate()

  report_section = get_text_section(stdoutdata, "Detailed Accuracy By Class")

  num_classes = 0
  total_precision = 0.0
  total_recall = 0.0
  for report_line in report_section.splitlines():
    if is_class_report_line(report_line):
      components = report_line.split()
      num_classes += 1
      total_precision += float(components[2])
      total_recall += float(components[3])

  precision = total_precision/num_classes
  recall = total_recall/num_classes
  f1 = 2*(precision*recall)/(precision+recall)

  report = {'f1': f1, 'precision': precision, 'recall': recall}

  print(json.dumps(report), end='')
