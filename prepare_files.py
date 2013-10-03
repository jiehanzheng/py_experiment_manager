from __future__ import print_function
import sys
import random
from collections import namedtuple, defaultdict, Counter
import itertools
import os


class Example():
  def __init__(self, line_num, klass, example_id, fold):
    self.line_num = line_num
    self.klass = klass
    self.example_id = example_id
    self.fold = fold


def get_examples_indexed_by_class(input_file):
  examples = defaultdict(list)

  for line_num, input_line in enumerate(input_file):
    example_id, klass = input_line.split(" ", 3)[:2]
    examples[klass].append(Example(line_num=line_num, klass=klass, example_id=example_id, fold=None))
  
  return examples


def get_total_number_of_examples(examples):
  num_examples = 0
  for examples_in_class in examples.values():
    num_examples = num_examples + len(examples_in_class)

  return num_examples


def get_line_to_example_with_fold_attribute_mapping(examples, num_folds):
  line_to_example = [None] * get_total_number_of_examples(examples)

  # shuffle within each class
  i = -1
  for examples_in_class in examples.values():
    random.shuffle(examples_in_class)

    # assign fold number to examples in each class
    for example in examples_in_class:
      i = i + 1
      example.fold = i % num_folds + 1
      line_to_example[example.line_num] = example

  return line_to_example


def get_folded_file_name(input_file_name, fold_id):
  return input_file_name + '.fold_' + str(fold_id)


def get_fold_folder_name(input_file_name, train_folds):
  return input_file_name + "_model_" + '_'.join([str(fold_id) for fold_id in train_folds])


if __name__ == "__main__":
  assert len(sys.argv) == 3, "need two args"
  input_file_name = sys.argv[1]
  num_folds = int(sys.argv[2])
  assert num_folds > 0, "invalid num_folds"

  expected_folds_filename = os.path.basename(input_file_name) + '.' + str(num_folds) + '_folds'

  with open(input_file_name) as input_file:
    examples = get_examples_indexed_by_class(input_file)

  num_classes = len(examples)

  # check if folds file exists
  if not os.path.exists(expected_folds_filename):
    print("Folds file do not exist, creating one...")

    print("Number of classes:", num_classes, file=sys.stderr)
    print("Number of examples:", get_total_number_of_examples(examples), file=sys.stderr)

    line_to_example = get_line_to_example_with_fold_attribute_mapping(examples, num_folds)
    
    # write to file, meanwhile, gather distribution information
    counts_indexed_by_folds = defaultdict(Counter)
    with open(expected_folds_filename, 'w') as folds_file:
      for example in line_to_example:
        folds_file.write(str(example.example_id) + '\t' + str(example.fold) + '\n')
        counts_indexed_by_folds[example.fold][example.klass] += 1

    print("Actual distribution:", counts_indexed_by_folds, file=sys.stderr)
  else:
    print("Using existing folds file:", expected_folds_filename, file=sys.stderr)

  # start distributing examples to folds
  folded_files = dict()
  for fold_id in range(1, num_folds + 1):
    folded_files[str(fold_id)] = open(get_folded_file_name(input_file_name, fold_id), 'w')

  with open(expected_folds_filename) as folds_file, open(input_file_name) as input_file:
    for folds_line, input_line in zip(folds_file, input_file):
      id_in_folds, target_fold = folds_line.strip().split('\t')
      id_in_input = input_line.split(' ')[0]
      rest_of_input_line = ' '.join(input_line.split(' ')[1:])

      assert id_in_folds == id_in_input

      folded_files[target_fold].write(rest_of_input_line)

  for fold_id in range(1, num_folds + 1):
    folded_files[str(fold_id)].close()

  # create train/test combinations
  for train_folds in itertools.combinations(range(1, num_folds+1), num_folds-1):
    folder_name = get_fold_folder_name(input_file_name, train_folds)

    print("Creating folds folder:", folder_name, file=sys.stderr)
    try:
      os.makedirs(folder_name)
    except OSError:
      print("Folds folder already exists.", file=sys.stderr)

    # concat corresponding train & test files
    with open(os.path.join(folder_name, 'train'), 'w') as train_file:
      for fold_id in train_folds:
        with open(get_folded_file_name(input_file_name, fold_id)) as input_file:
          print("Copying", input_file.name, "to", train_file.name, file=sys.stderr)
          for input_line in input_file:
            train_file.write(input_line)

    # figure out our test set then copy test files
    test_folds = set(range(1, num_folds+1)) - set(train_folds)
    with open(os.path.join(folder_name, 'test'), 'w') as test_file:
      for fold_id in test_folds:
        with open(get_folded_file_name(input_file_name, fold_id)) as input_file:
          print("Copying", input_file.name, "to", test_file.name, file=sys.stderr)
          for input_line in input_file:
            test_file.write(input_line)

  # remove fold files to save space
  for fold_id in range(1, num_folds + 1):
    os.remove(folded_files[str(fold_id)].name)
