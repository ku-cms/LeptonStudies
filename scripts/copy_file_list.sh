#!/usr/bin/bash
# copy_file_list.sh

# Copies all files in a file list to an output directory using xrdcp.

# Requires CERN grid certificate and active voms proxy:
# voms-proxy-init --valid 192:00 -voms cms
# voms-proxy-info

# To run:

# User inputs:
# - file list
# - output directory

# Output:
# - copies files to output directory

# User inputs:
FILE_LIST=$1
OUTPUT_DIR=$2

# Main FNAL redirector (searches all sites):
REDIRECTOR=root://cmsxrootd.fnal.gov/

echo "FILE_LIST: ${FILE_LIST}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "REDIRECTOR: ${REDIRECTOR}"

# Check if FILE_LIST is empty.
if [[ -z "$FILE_LIST" ]]; then
    echo "ERROR: FILE_LIST is empty."
    echo "Please provide a file list as the first argument."
    exit 1
fi

# Check if OUTPUT_DIR is empty.
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: OUTPUT_DIR is empty."
    echo "Please provide an ouput directory as the second argument."
    exit 1
fi

# Check if file list does not exist.
if [[ ! -f "$FILE_LIST" ]]; then
    echo "ERROR: The file '${FILE_LIST}' does not exist!"
    exit 1
fi

# Create output directory.
mkdir -p ${OUTPUT_DIR}

echo "Copying files using xrdcp..."

while read line; do
    echo "${line}"
done < ${FILE_LIST}

# print number of files copied

echo "Done!"


