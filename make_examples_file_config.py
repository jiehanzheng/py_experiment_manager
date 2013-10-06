from ConfigParser import SafeConfigParser

if __name__ == "__main__":
  config = SafeConfigParser()

  config.add_section('screenplay')
  config.set('screenplay', 'examples_file', 'screenplay.arff')
  config.set('screenplay', 'num_folds', '5')
  config.set('screenplay', 'num_test_folds', '1')

  config.write(open('examples_file.cfg', 'w'))
