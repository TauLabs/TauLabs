import uavo

class UAVOCollection(dict):
    def __init__(self):
        self.clear()

    def find_by_name(self, uavo_name):
        for u in self.itervalues():
            if u.meta['name'] == uavo_name:
                return u
        return None

    def from_git_hash(self, githash):
        import subprocess
        import tarfile
        from cStringIO import StringIO
        #
        # Grab the exact uavo definition files from the git repo using the header's git hash
        #
        p = subprocess.Popen(['git', 'archive', githash, '--', 'shared/uavobjectdefinition/'],
                             stdout=subprocess.PIPE)
        # grab the tar file data
        git_archive_data, git_archive_errors = p.communicate()

        # coerce the tar file data into a file object so that tarfile likes it
        fobj = StringIO(git_archive_data)

        # feed the tar file data to a tarfile object
        t = tarfile.open(fileobj=fobj)

        # Build up the uavo definitions for all of the available UAVO at this git hash
        for f_info in t.getmembers():
            if not f_info.isfile():
                continue

            f = t.extractfile(f_info)

            u = uavo.UAVO()
            u.from_xml(f)

            # add this uavo definition to our dictionary
            self.update([('{0:08x}'.format(u.id), u)])

    def from_uavo_xml_path(self, path):
        import os
        import glob

        for file_name in glob.glob(os.path.join(path, '*.xml')):
            with open(file_name, 'rU') as f:
                u = uavo.UAVO()
                u.from_xml(f)

                # add this uavo definition to our dictionary
                self.update([('{0:08x}'.format(u.id), u)])


