from ConfigParser import SafeConfigParser

if __name__ == "__main__":
  config = SafeConfigParser({'username': 'jzheng', 'num_concurrent_jobs': 4, 'working_directory': '/localtemp/jzheng/screenplay'})

  config.add_section('ghazali')
  config.set('ghazali', 'hostname', 'ghazali.ldeo.columbia.edu')

  config.add_section('pupin')
  config.set('pupin', 'hostname', 'pupin.ldeo.columbia.edu')

  config.write(open('servers.cfg', 'w'))
