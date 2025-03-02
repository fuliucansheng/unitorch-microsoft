# Setup A6000 Environment

### Open the firewall ports
```bash
sudo apt install iptables-persistent
sudo iptables -P INPUT ACCEPT
sudo iptables -P FORWARD ACCEPT
sudo iptables -P OUTPUT ACCEPT
sudo iptables -F
sudo iptables-save
sudo netfilter-persistent save
sudo netfilter-persistent reload
```

### Setup unitorch-microsoft

```bash
mkdir -p ~/my
cd ~/my
git clone https://dev.azure.com/decui/unitorch-microsoft/_git/unitorch-microsoft && cd unitorch-microsoft && pip3 install -e .
pip3 install gradio==4.40.0 fastapi
```

### Start the flux model fastapi service

> You could test the service by visiting the URL like: http://br1t44-s3-17:5000/docs

```bash
unitorch-fastapi omnipixel/configs/tools/fastapis.ini --device 0 --port 5000
```

### Run the client

> API Endpoint should be like: http://br1t44-s3-17:5000/core/fastapi/stable/text2image

```bash
unitorch-launch omnipixel/scripts/text2image.ini --data_file ./data.tsv --jsonl_file ./output.jsonl
```