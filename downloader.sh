#!/bin/bash

# Description:
#     Download the CTU-13 dataset from https://www.stratosphereips.org/datasets-ctu13


# Usage
#     $ bash downloader.sh

function Downloader(){
    # Download the .tar.bz of the CTU-13 
    wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2 --no-check-certificate

    tar -xf CTU-13-Dataset.tar.bz2
}

function Cleanup(){
    # remove archive
    rm -f CTU-13-Dataset.tar.bz2
}

function Main(){
    Downloader
    Cleanup
}

Main
exit 0