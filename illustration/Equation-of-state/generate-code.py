import os, git, sys
import numpy as np

# detect the path to the documentation
current_path = os.getcwd()
git_repo = git.Repo(current_path, search_parent_directories=True)
git_path = git_repo.git.rev_parse("--show-toplevel")
path_to_docs = git_path + "/mdcourse.github.io/docs/source/chapters/"

# import the python converter
path_to_converter = git_path + "/mdcourse.github.io/tests/"
sys.path.append(path_to_converter)
from utilities import sphinx_to_python

# make sure the documentation was found
assert os.path.exists(path_to_docs), """Documentation files not found"""

# if necessary, create the "generated-codes/" folder
if os.path.exists("generated-codes/") is False:
    os.mkdir("generated-codes/")

# Choose a desired chapter id to build the code from
# max_chapter_id = 6
for chapter_id in [1, 2, 3, 4, 5, 6]:
    RST_EXISTS, created_tests, folder = sphinx_to_python(path_to_docs, chapter_id)
