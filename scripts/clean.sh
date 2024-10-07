#! /bin/bash

# Clean up data directory.
for file in data/*.csv
do
    rm $file
done

exit 0
