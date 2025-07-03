#!/usr/bin/env bash
# Использует sample_pairs.tsv и заново строит:
#   ▸ bw/${atac_id}_ATAC.bw
#   ▸ peaks/${chip_id}/*   +  bw/${chip_id}_FE.bw
# Перед запуском ПОЛНОСТЬЮ чистит каталоги bw/ peaks/

set -euo pipefail
ROOT="$(dirname "$(dirname "$0")")"     # …/ChipSeqAtacBAMS
cd "$ROOT"

rm -rf bw peaks
mkdir  bw peaks

# читаем sample_pairs.tsv (пропускаем заголовок)
tail -n +2 sample_pairs.tsv | while IFS=$'\t' read -r cid chip_id chip_bam atac_id atac_bam; do
    echo "▶ $cid  ($chip_id  /  $atac_id)"

    # ---------- ATAC  (bamCoverage) ----------
    out_bw="bw/${atac_id}_ATAC.bw"
    bamCoverage -b "$atac_bam" -o "$out_bw" \
                --extendReads --binSize 10 --normalizeUsing CPM

    # ---------- ChIP + FE ----------
    pico="peaks/${chip_id}"
    macs3 callpeak -t "$chip_bam" -f BAM -g hs -n "$chip_id" \
          --outdir "$pico" --keep-dup all --bdg --call-summits -q 0.01

    macs3 bdgcmp -t "$pico/${chip_id}_treat_pileup.bdg" \
                 -c "$pico/${chip_id}_control_lambda.bdg" \
                 --method FE -o "$pico/FE.bdg"

    sort -k1,1 -k2,2n "$pico/FE.bdg" > "$pico/FE.sorted.bdg"
    bedGraphToBigWig "$pico/FE.sorted.bdg" hg38.chrom.sizes \
                     "bw/${chip_id}_FE.bw"

    echo "✓  $cid готов"
done
