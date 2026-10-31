#!/usr/bin/env bash
# Install INetSim on the gateway and bind it to the sandbox network only.
set -eu
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq inetsim

cat > /etc/inetsim/inetsim.conf <<'EOF'
service_bind_address 192.168.100.1
start_service dns
start_service http
start_service https
start_service smtp
start_service smtps
start_service pop3
start_service pop3s
start_service ftp
start_service ntp
start_service irc
dns_default_ip 192.168.100.1
dns_bind_port 53
EOF

# Never forward packets, and keep SSH (vagrant/vagrant) off the sandbox network.
echo 'net.ipv4.ip_forward = 0' > /etc/sysctl.d/99-no-forward.conf
sysctl -q --system
systemctl enable inetsim
systemctl disable ssh
