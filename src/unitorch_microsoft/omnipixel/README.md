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
```