#!/usr/bin/env bash
wget "https://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip" -P "data/IHDP"
wget "https://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip" -P "data/IHDP"
unzip "data/IHDP/ihdp_npci_1-1000.train.npz.zip" -d "data/IHDP/"
unzip "data/IHDP/ihdp_npci_1-1000.test.npz.zip" -d "data/IHDP/"



