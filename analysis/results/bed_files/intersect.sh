#!/bin/bash

MOTIF_FILE=$1
OUTFILE=$2

bedtools intersect -a $MOTIF_FILE -b 5ae.bed -f 1 -wo > $OUTFILE/ae.bed
bedtools intersect -a $MOTIF_FILE -b 5peads.bed -f 1 -wo > $OUTFILE/peads.bed
bedtools intersect -a $MOTIF_FILE -b 5peas.bed -f 1 -wo > $OUTFILE/peas.bed