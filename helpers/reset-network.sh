#!/bin/sh

### RESET NETWORK PERFORMANCE TO DEFAULT ###
sudo sysctl -w net.core.rmem_default=212992
sudo sysctl -w net.core.rmem_max=212992
sudo sysctl -w net.core.wmem_default=212992
sudo sysctl -w net.core.wmem_max=212992
sudo sysctl -w net.ipv4.tcp_rmem="4096 131072 6291456"
sudo sysctl -w net.ipv4.tcp_wmem="4096 16384 4194304"
sudo sysctl -w net.ipv4.tcp_mem="190878 254505 381756"
sudo sysctl -w net.ipv4.udp_mem="381756 509011 763512"
sudo sysctl -w net.ipv4.udp_rmem_min=4096
sudo sysctl -w net.ipv4.udp_wmem_min=4096