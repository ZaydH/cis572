#!/usr/bin/env bash
SRC_FILE=scatter_sep_data.csv
OUT_FILE=compressed_${SRC_FILE}
awk 'NR == 1 || NR % 3 == 0' ${SRC_FILE} > ${OUT_FILE}
