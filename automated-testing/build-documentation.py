#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, git
import shutil
import subprocess

current_path = os.getcwd()
git_repo = git.Repo(current_path, search_parent_directories=True)
git_path = git_repo.git.rev_parse("--show-toplevel")

sys.path.append(git_path + "/functions/")

path_to_docs = git_path + "/mdcourse.github.io/docs/source/chapters/"

assert os.path.exists(path_to_docs), """Documentation files not found"""


# In[9]:


# In chapter 1, all the files are simply created.
chapter_id = 1
filename = path_to_docs + "chapter"+str(chapter_id)+".rst"
test_files = []
list_files = []
if os.path.exists(filename):
    # saving folder
    folder = "generated-codes/chapter"+str(chapter_id)+"/"
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    file = open(filename, "r")
    print_file = False
    for line in file: # Loop over all the lines of the file
        if ".. label::" in line: # Detect the label "start" and label "end"
            label = line.split(".. label:: ")[1] # Look for label in the line
            if label[:6] == "start_": # Detect starting label
                class_name_i = label.split("start_")[1].split("_class")[0]
                if "test" in class_name_i:
                    test_files.append(class_name_i+".py")
                else:
                    list_files.append(class_name_i+".py")
                print_file = True
                myclass = open(folder+class_name_i+".py", "w")
            elif label[:4] == "end_": # Detect ending label
                class_name_f = label.split("end_")[1].split("_class")[0]
                assert class_name_f == class_name_i, """Different class closed, inconsistency in rst file?"""
                print_file = False
                myclass.close()
        else:
            if print_file: # Print the content of the label into files
                if ".. code-block::" not in line: # Ignore code block line
                    if len(line) > 1: # Remove the indentation
                        myclass.write(line[4:])
                    else:
                        myclass.write(line)
# Test
mycwd = os.getcwd()
os.chdir(folder)
for test_file in test_files:
    subprocess.call(["python3", test_file])
os.chdir(mycwd)


# In[18]:


chapter_id = 2
filename = path_to_docs + "chapter"+str(chapter_id)+".rst"
test_files = []
if os.path.exists(filename):
    # saving folder
    folder = "generated-codes/chapter"+str(chapter_id)+"/"
    previous_folder = "generated-codes/chapter"+str(chapter_id-1)+"/"
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    # copy all the files from the previous chapter
    for file in list_files:
        shutil.copyfile(previous_folder+"/"+file, folder+"/"+file)
    file = open(filename, "r")
    print_file = False
    for line in file: # Loop over all the lines of the file
        if ".. label::" in line: # Detect the label "start" and label "end"
            label = line.split(".. label:: ")[1] # Look for label in the line
            if label[:6] == "start_": # Detect starting label
                class_name_i = label.split("start_")[1].split("_class")[0]
                if "test" in class_name_i:
                    test_files.append(class_name_i+".py")
                print_file = True
                myclass = open(folder+class_name_i+".py", "a")
            elif label[:4] == "end_": # Detect ending label
                class_name_f = label.split("end_")[1].split("_class")[0]
                assert class_name_f == class_name_i, """Different class closed, inconsistency in rst file?"""
                print_file = False
                myclass.close()
        else:
            if print_file: # Print the content of the label into files
                if ".. code-block::" not in line: # Ignore code block line
                    if len(line) > 1: # Remove the indentation
                        myclass.write(line[4:])
                    else:
                        myclass.write(line)


# In[12]:






# In[ ]:


class_list = []
for chapter_id in range(100):
    filename = path_to_docs + "chapter"+str(chapter_id)+".rst"
    if os.path.exists(filename):
        # saving folder
        folder = "generated-codes/chapter"+str(chapter_id)+"/"
        if os.path.exists(folder) is False:
            os.mkdir(folder)
        file = open(filename, "r")
        print_file = False
        for line in file:
            if ".. label::" in line:
                # Detect the label "start" and label "end"
                label = line.split(".. label:: ")[1]
                if label[:6] == "start_":
                    class_name = label.split("start_")[1].split("_class")[0]
                    class_list.append([chapter_id, class_name])
                    print_file = True
                    myclass = open(folder+class_name+".py", "w")
                elif label[:4] == "end_":
                    class_name = label.split("end_")[1].split("_class")[0]
                    print_file = False
                    myclass.close()
            else:
                # Print the content of the label into files
                if print_file:
                    if ".. code-block::" not in line:
                        if len(line) > 1:
                            myclass.write(line[4:])
                        else:
                            myclass.write(line)


# In[ ]:





# In[ ]:


if chapter_id > 1:
    for my_id, class_name in class_list:
        if "test_" not in class_name:
            print("generated-codes/chapter"+str(my_id)+"/"+class_name+".py")
            shutil.copyfile("generated-codes/chapter"+str(my_id)+"/"+class_name+".py",
                            "generated-codes/chapter"+str(chapter_id)+"/"+class_name+".py")


# In[ ]:





# In[ ]:





