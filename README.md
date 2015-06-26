# Winpycaffe

- this is a windows port of caffe including msvc build files for pycaffe
- additionally, this repository contains pull requests #1976, #2016 and #2086
- tested with python 2.7 amd64, cuda 7.0 and msvc12
- actually, the latest version hasn't been fully tested yet
- build files will automatically download 3rd party binary package from my dropbox
- this port is based on https://github.com/niuzhiheng/caffe
- putting missing functions for windows into a 3rd party package gives minimal changes in code and easier maintenance, see https://github.com/MalteOeljeklaus/libwincaffe_misc

# Caffe
This is a pre-release Caffe branch for fully convolutional networks. This includes unmerged PRs and no guarantees.

Everything here is subject to change, including the history of this branch.

Consider PR #2016 for reducing memory usage.

See `future.sh` for details.
