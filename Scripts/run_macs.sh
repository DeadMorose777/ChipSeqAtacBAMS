#!/usr/bin/env bash
# Используется GNU parallel’ем. Четыре аргумента приходят из sample_table.tsv:
#   $1 – bam-файл; $2 – assay (ChIP или ATAC); $3 – sample_id; $4 – control_bam или NA
set -euo pipefail

bam="$1"
assay="$2"
name="$3"
ctrl="$4"

ROOT=$(dirname "$(dirname "$0")")          # /mnt/d/ChipSeqAtacBAMS
PEAKS="$ROOT/peaks"
BW="$ROOT/bw"
mkdir -p "$PEAKS" "$BW"

if [[ "$assay" == "ChIP" ]]; then
  # ------------------------------------------------------- #
  #  Поддержка ChIP-экспериментов без control-BAM
  # ------------------------------------------------------- #
  if [[ "$ctrl" == "NA" || -z "$ctrl" ]]; then
    CTRL_ARG=()
    echo "⚠️  $name без control-BAM — вызываем MACS3 без опции -c"
  else
    CTRL_ARG=(-c "$ctrl")
  fi

  macs3 callpeak \
        -t "$bam" "${CTRL_ARG[@]}" \
        -f BAM -g hs \
        -n "$name" --outdir "$PEAKS/$name" \
        --keep-dup all --bdg --call-summits -q 0.01

  # FE треки → BigWig
  macs3 bdgcmp \
        -t "$PEAKS/$name/${name}_treat_pileup.bdg" \
        -c "$PEAKS/$name/${name}_control_lambda.bdg" \
        --method FE -o "$PEAKS/$name/FE.bdg"

  sort -k1,1 -k2,2n "$PEAKS/$name/FE.bdg" > "$PEAKS/$name/FE.sorted.bdg"
  bedGraphToBigWig \
        "$PEAKS/$name/FE.sorted.bdg" \
        "$ROOT/hg38.chrom.sizes" \
        "$BW/${name}_FE.bw"

else   # ---------- ATAC ---------------------------------------------------- #
  bamCoverage \
        -b "$bam" \
        -o "$BW/${name}_ATAC.bw" \
        --extendReads --binSize 10 --normalizeUsing CPM
fi

echo "✓ обработан $name"
