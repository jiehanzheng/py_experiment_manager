from __future__ import print_function
from .runner import Runner
from paramiko import SSHClient, SFTPClient, AutoAddPolicy
from pipes import quote
import sys
import os
import tarfile
import subprocess
import hashlib
import logging
import json
import uuid


class SSHRunner(Runner):

  environment_variables = set()
  ssh = None
  ssh_closed = True
  bin_path = ""

  def __init__(self, username, hostname, working_directory, logger=None):
    self.username = username
    self.hostname = hostname

    if logger is None:
      self.logger = logging.getLogger()
    else:
      self.logger = logger

    self.logger.info("Connecting to " + username + "@" + hostname + "...")

    # try connecting to it
    self._ensure_connected()

    _, home, _ = self.ssh.exec_command("pwd")
    home = home.read().strip()
    self.bin_path = os.path.join(home, 'bin')
    self.working_directory = working_directory.replace('~', home)

    # add ~/bin to path if it is not already there
    _, path, _ = self.ssh.exec_command("echo $PATH")
    path_string = path.read()
    if home + '/bin' not in path_string.split(':'):
      self.environment_variables.add(('PATH', self.bin_path + ':$PATH'))

    # check if SVMLight is installed
    _, svmlearn_location, _ = self._run_command("which svm_learn")
    svmlearn_location = svmlearn_location.read().strip()

    if len(svmlearn_location) > 0:
      self.logger.info(hostname + " has SVMLight installed at " + svmlearn_location)
    else:
      self.logger.warning(hostname + " does not have SVMLight installed.  Attempting to install one...")
      
      self._run_command("mkdir -p " + self.bin_path)

      # download and install SVMLight to home bin
      self._run_command('wget -O ' + quote(os.path.join(self.bin_path, 'svm_learn')) + ' ' + 'https://static.jiehan.org/pub/svm_light/svm_learn')
      self._run_command('wget -O ' + quote(os.path.join(self.bin_path, 'svm_classify')) + ' ' + 'https://static.jiehan.org/pub/svm_light/svm_classify')
      self._run_command("chmod u+x " + quote(os.path.join(self.bin_path, 'svm_learn')))
      self._run_command("chmod u+x " + quote(os.path.join(self.bin_path, 'svm_classify')))

    # check if our server-side script is installed
    install_helper_script = True

    local_script_location = os.path.join(os.path.dirname(__file__), '../standalone_scripts/train_and_test.py')

    _, script_location, _ = self._run_command("which train_and_test.py")
    script_location = script_location.read().strip()

    if len(script_location) > 0:
      # check version
      _, script_md5_line, _ = self._run_command("md5sum " + quote(script_location))
      remote_script_md5, _ = script_md5_line.read().strip().split()[:2]
      local_script_md5 = hashlib.md5(open(local_script_location, 'rb').read()).hexdigest()

      if remote_script_md5 == local_script_md5:
        install_helper_script = False
        self.logger.info(hostname + "'s helper script at " + script_location + " is up-to-date.")
      else:
        install_helper_script = True
        self.logger.info(hostname + "'s copy of helper script is out-of-date (remote=" + remote_script_md5 + ', local=' + local_script_md5 + ')')
    else:
      self.logger.info(hostname + " does not have helper script installed.")

    if install_helper_script:
      self.logger.info("Copying helper script to " + hostname + '...')
      self._copy_to_server(local_script_location, self.bin_path)
      self._run_command("chmod u+x " + quote(os.path.join(self.bin_path, '/train_and_test.py')))

    # create working directory
    self._run_command("rm -rf " + quote(self.working_directory))
    self._run_command("mkdir " + quote(self.working_directory))

    self.ssh.close()
    self.ssh_closed = True


  def logger(self):
    return logger

  def _ensure_connected(self):
    if self.ssh_closed:
      self.ssh = SSHClient()
      self.ssh.load_system_host_keys()
      self.ssh.set_missing_host_key_policy(AutoAddPolicy())
      self.ssh.connect(self.hostname, username=self.username)
      self.ssh.get_transport().set_keepalive(30)
      self.ssh_closed = False


  def _run_command(self, command):
    self._ensure_connected()
    if self.environment_variables is not None:
      command = '; '.join(['export ' + e[0] + '=' + e[1] for e in self.environment_variables]) + '; ' + command

    self.logger.debug("SSH: " + command)
    return self.ssh.exec_command(command)

  def do_experiment(self, folder_name, svm_params=None):
    if svm_params is None:
      svm_params = ""

    self._ensure_connected()
    self.logger.info(self.hostname + " is assigned " + folder_name + '.')

    # copy folder over
    self._copy_to_server(folder_name)

    self.logger.info(self.hostname + " is training on " + folder_name + '...')
    _, stdout, stderr = self._run_command('cd ' + self.working_directory + '/' + folder_name + '; ' + 'train_and_test.py ' + quote(svm_params))

    json_result = stdout.read()

    if len(json_result) == 0:
      error_message = "Unexpected output from remote server.  STDERR:\n" + stderr.read()
      self.logger.critical(error_message)
      raise RuntimeError(error_message)

    self._cleanup(folder_name)

    # maybe this will help solving the getting stuck problem 
    self.ssh.close()
    self.ssh_closed = True

    try:
      return json.loads(json_result)
    except ValueError:
      raise RuntimeError("Unable to parse STDOUT with JSON parser:\n" + json_result)

  def _copy_to_server(self, filename, destination=None):
    self._ensure_connected()

    if destination is None:
      destination = self.working_directory

    sftp = SFTPClient.from_transport(self.ssh.get_transport())

    # tar and compress a directory before copying
    if os.path.isfile(filename):
      sftp.put(filename, os.path.join(destination, os.path.basename(filename)))
    elif os.path.isdir(filename):
      tar_filename = filename + '_' + str(uuid.uuid4()) + '.tar.gz'

      self.logger.info("Compressing " + filename + '...')
      subprocess.call(['tar', 'cfz', tar_filename, filename])

      self.logger.info("Copying " + tar_filename + " to " + self.hostname + ':' + os.path.join(destination, os.path.basename(tar_filename)) + '...')
      sftp.put(tar_filename, os.path.join(destination, os.path.basename(tar_filename)))
      os.remove(tar_filename)

      self._run_command('cd ' + self.working_directory + '; tar xfz ' + quote(tar_filename) + '; rm ' + quote(tar_filename))
    else:
      raise NotImplementedError("Can't SFTP put anything other than a folder or file yet.")

    sftp.close()

  def _cleanup(self, folder_name=''):
    self._ensure_connected()
    self._run_command("rm -rf " + quote(self.working_directory) + '/' + quote(folder_name))
