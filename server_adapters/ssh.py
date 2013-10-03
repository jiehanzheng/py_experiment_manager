from __future__ import print_function
from .runner import Runner
from paramiko import SSHClient, SFTPClient
from pipes import quote
import sys
import os
import tarfile
import subprocess
import hashlib


class SSHRunner(Runner):

  environment_variables = set()
  ssh = None
  bin_path = ""

  def __init__(self, username, hostname, working_directory):
    self.username = username
    self.hostname = hostname

    print("Connecting to", username + "@" + hostname + "...", file=sys.stderr)

    # try connecting to it
    self.ssh = SSHClient()
    self.ssh.load_system_host_keys()
    self.ssh.connect(hostname, username=username)

    _, home, _ = self.ssh.exec_command("pwd")
    home = home.read().strip()
    self.bin_path = quote(home) + '/bin'
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
      print(hostname, "has SVMLight installed at", svmlearn_location, file=sys.stderr)
    else:
      print(hostname, "does not have SVMLight installed.  Attempting to install one...", file=sys.stderr)
      
      self._run_command("mkdir -p " + self.bin_path)

      # download and install SVMLight to home bin
      # self._run_command("mkdir -p /tmp/py_experiment_manager")
      # self._run_command("wget -O /tmp/py_experiment_manager/svmlight.tgz http://download.joachims.org/svm_light/current/svm_light_linux64.tar.gz")
      # self._run_command("mkdir -p /tmp/py_experiment_manager/svmlight")
      # self._run_command("tar xfz /tmp/py_experiment_manager/svmlight.tgz -C /tmp/py_experiment_manager/svmlight")
      # self._run_command("mv /tmp/py_experiment_manager/svmlight/svm_learn /tmp/py_experiment_manager/svmlight/svm_classify " + self.bin_path)
      # self._run_command("rm -rf /tmp/py_experiment_manager/")
      self._run_command('wget -O ' + quote(self.bin_path) + '/svm_learn ' + 'https://static.jiehan.org/pub/svm_light/svm_learn')
      self._run_command('wget -O ' + quote(self.bin_path) + '/svm_classify ' + 'https://static.jiehan.org/pub/svm_light/svm_classify')
      self._run_command("chmod u+x " + quote(self.bin_path) + '/svm_learn')
      self._run_command("chmod u+x " + quote(self.bin_path) + '/svm_classify')

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
        print(hostname, "'s helper script at", script_location, "is up-to-date.", file=sys.stderr)
      else:
        install_helper_script = True
        print(hostname + "'s copy of helper script is out-of-date (remote=" + remote_script_md5 + ', local=' + local_script_md5 + ')', file=sys.stderr)
    else:
      print(hostname, "does not have helper script installed.", file=sys.stderr)

    if install_helper_script:
      print("Copying helper script to", hostname, file=sys.stderr)
      self._copy_to_server(local_script_location, self.bin_path)
      self._run_command("chmod u+x " + quote(self.bin_path) + '/train_and_test.py')

    # create working directory
    self._run_command("mkdir " + working_directory)


  def _run_command(self, command):
    if self.environment_variables is not None:
      command = '; '.join(['export ' + e[0] + '=' + e[1] for e in self.environment_variables]) + '; ' + command

    return self.ssh.exec_command(command)

  def do_experiment(self, folder_name):
    print(self.hostname, "is working on", folder_name + '...', file=sys.stderr)

    # copy folder over
    self._copy_to_server(folder_name)
    _, stdout, stderr = self._run_command('cd ' + self.working_directory + '/' + folder_name + '; ' + 'train_and_test.py')

    json_result = stdout.read()

    if len(json_result) == 0:
      raise RuntimeError("Unexpected output from remote server.  STDERR:\n" + stderr.read())

    return json_result

  def _copy_to_server(self, filename, destination=None):
    if destination is None:
      destination = self.working_directory

    sftp = SFTPClient.from_transport(self.ssh.get_transport())

    # tar and compress a directory before copying
    if os.path.isfile(filename):
      sftp.put(filename, os.path.join(destination, os.path.basename(filename)))
    elif os.path.isdir(filename):
      tar_filename = filename + '.tar.gz'

      print("Compressing", filename + '...', file=sys.stderr)
      subprocess.call(['tar', 'cfz', tar_filename, filename])

      print("Copying", tar_filename, "to", self.hostname + ':' + os.path.join(destination, os.path.basename(tar_filename)))
      sftp.put(tar_filename, os.path.join(destination, os.path.basename(tar_filename)))
      os.remove(tar_filename)

      self._run_command('cd ' + self.working_directory + '; tar xfz ' + tar_filename)
    else:
      raise NotImplementedError("Can't SFTP put anything other than a folder or file yet.")

    sftp.close()

  def _cleanup():
    self._run_command("rm -rf " + quote(working_directory))
