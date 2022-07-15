import pkg_resources
from subprocess import call

if __name__ == '__main__':
    packages = [dist.project_name for dist in pkg_resources.working_set]
    call("pip3 install --upgrade " + ' '.join(packages), shell=True)
