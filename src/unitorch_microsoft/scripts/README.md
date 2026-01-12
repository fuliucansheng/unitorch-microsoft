# Script Tools

### Hf Hub

##### Upload File
```bash
python3.10 -m unitorch_microsoft.scripts.tools.hfhub upload_file --repo fuliucansheng/tempfiles --repo_type dataset --local_file ./badcrop_ta/d41b675f7dd1a7ce527b607a0d867f69.jpg --remote_file badcrop_ta/d41b675f7dd1a7ce527b607a0d867f69.jpg
```

##### Upload Folder
```bash
python3.10 -m unitorch_microsoft.scripts.tools.hfhub upload_folder --repo fuliucansheng/tempfiles --repo_type dataset --local_folder ./badcrop_ta --remote_folder badcrop_ta
```

##### Delete File
```bash
python3.10 -m unitorch_microsoft.scripts.tools.hfhub delete_file --repo fuliucansheng/tempfiles --repo_type dataset --remote_file badcrop_ta/d41b675f7dd1a7ce527b607a0d867f69.jpg
```

##### Delete Folder
```bash
python3.10 -m unitorch_microsoft.scripts.tools.hfhub delete_folder --repo fuliucansheng/tempfiles --repo_type dataset --remote_folder badcrop_ta
```
