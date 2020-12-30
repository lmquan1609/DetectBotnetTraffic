#!/bin/bash

# Description:
#     Download the CTU-13 dataset from https://www.stratosphereips.org/datasets-ctu13


# Usage
#     $ bash downloader.sh

function Downloader(){
    # Download the .tar.bz of the CTU-13 
    !python download_gdrive.py 1XyDCTcmNIrQh_CFWCsqjR3RVyYUDKX7O CTU-13-Dataset.zip

    !unzip CTU-13-Dataset.zip
}

function Cleanup(){
    # remove archive
    rm -f CTU-13-Dataset.zip
}

function Main(){
    Downloader
    Cleanup
}

Main
exit 0