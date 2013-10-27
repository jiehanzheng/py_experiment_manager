from ConfigParser import SafeConfigParser

if __name__ == "__main__":
  config = SafeConfigParser({'svm_classifier': 'weka.classifiers.functions.SMO',
                             'svm_C': '1.0',
                             'svm_L': '0.001',
                             'svm_P': '1.0E-12',
                             'svm_N': '0',
                             'svm_V': '-1',
                             'svm_kernel_C': '250007',
                             'svm_kernel_E': '1.0'})

  config.add_section('c=10')
  config.set('c=10', 'svm_c', '10.0')

  config.add_section('c=100')
  config.set('c=100', 'svm_c', '100.0')

  config.write(open('experiments.cfg', 'w'))
