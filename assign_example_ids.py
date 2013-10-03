import sys
import os

def assign_ids(input_file, prefix):
  # create numbered train file
  with open(prefix + '.numbered', 'w') as numbered_file:
    example_id = -1
    for input_line in input_file:
      example_id = example_id + 1
      numbered_file.write(' '.join([prefix + '_' + str(example_id), input_line]))

if __name__ == "__main__":
  assert len(sys.argv) == 2, "need filename"
  input_file_name = sys.argv[1]

  with open(input_file_name) as input_file:
    assign_ids(input_file, os.path.basename(input_file_name))
