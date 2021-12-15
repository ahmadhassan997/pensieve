#!/bin/sh

### TUNING NETWORK PERFORMANCE ###
sudo sysctl -w net.core.rmem_default=31457280
sudo sysctl -w net.core.rmem_max=536870912
sudo sysctl -w net.core.wmem_default=31457280
sudo sysctl -w net.core.wmem_max=536870912
sudo sysctl -w net.ipv4.tcp_rmem="8192 16777216 536870912"
sudo sysctl -w net.ipv4.tcp_wmem="8192 16777216 536870912"
sudo sysctl -w net.ipv4.tcp_mem="786432 1048576 26777216"
sudo sysctl -w net.ipv4.udp_mem="65536 131072 262144"
sudo sysctl -w net.ipv4.udp_rmem_min=16384
sudo sysctl -w net.ipv4.udp_wmem_min=16384