#!/usr/bin/env bash
# $1 = bam_path  $2 = assay  $3 = sample_id  $4 = control_bam (может быть пустым)
set -euo pipefail
bam="$1"; assay="$2"; name="$3"; ctrl="$4"

ROOT=$(dirname "$(dirname "$0")")     # /mnt/d/ChipSeqAtacBAMS
PEAKS="$ROOT/peaks"; BW="$ROOT/bw";   mkdir -p "$PEAKS" "$BW"

# при необходимости индексируем
[[ -f "${bam}.bai" ]] || samtools index "$bam"
[[ -n "$ctrl" && ! -f "${ctrl}.bai" ]] && samtools index "$ctrl"

if [[ "$assay" == "ChIP" ]]; then
  # --- вызов MACS3 ---
  if [[ -n "$ctrl" ]]; then
      macs3 callpeak -t "$bam" -c "$ctrl" -f BAM -g hs \
            -n "$name" --outdir "$PEAKS/$name" \
            --keep-dup all --bdg --call-summits -q 0.01
  else
      macs3 callpeak -t "$bam"            -f BAM -g hs \
            -n "$name" --outdir "$PEAKS/$name" \
            --keep-dup all --bdg --call-summits -q 0.01
  fi

  # --- Fold-Enrichment → BigWig ---
  macs3 bdgcmp -t "$PEAKS/$name/${name}_treat_pileup.bdg" \
               -c "$PEAKS/$name/${name}_control_lambda.bdg" \
               --method FE -o "$PEAKS/$name/FE.bdg"
  sort -k1,1 -k2,2n "$PEAKS/$name/FE.bdg" \
      > "$PEAKS/$name/FE.sorted.bdg"
  bedGraphToBigWig "$PEAKS/$name/FE.sorted.bdg" \
      "$ROOT/hg38.chrom.sizes" "$BW/${name}_FE.bw"

else   # --- ATAC ---
  bamCoverage -b "$bam" -o "$BW/${name}_ATAC.bw" \
              --extendReads --binSize 10 --normalizeUsing CPM
fi

echo "✓ обработан $name"
