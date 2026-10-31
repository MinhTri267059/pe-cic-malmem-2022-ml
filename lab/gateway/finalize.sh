#!/usr/bin/env bash
# Cut the gateway off from the internet: no MTA, no SSH, no NAT adapter.
# Afterwards start it with: VBoxManage startvm Sandbox_GW --type headless
set -eu
cd "$(dirname "$0")"

vagrant ssh -c 'sudo systemctl disable --now exim4 postfix 2>/dev/null || true; sudo systemctl disable ssh'
vagrant halt
VBoxManage modifyvm Sandbox_GW --nic1 none
VBoxManage snapshot Sandbox_GW take clean_gateway --description "INetSim on sbxnet, no uplink"
VBoxManage showvminfo Sandbox_GW --machinereadable | grep -E '^(nic[1-8]|intnet)'
