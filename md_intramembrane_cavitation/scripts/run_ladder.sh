#!/bin/bash
# Direct test at the PRX 2014 operating point and above: hold the equilibrated
# bilayer at a fixed external normal pressure (rarefaction) and watch whether an
# intramembrane gap opens, and where any void nucleates (water vs membrane core).
#
# usage: run_ladder.sh <equilibrated.cfg> <outdir> <steps> <pnorm values (reduced)...>
set -u
CFG=$1; OUT=$2; STEPS=$3; shift 3
BIN="$(dirname "$0")/../src/cgmd"
mkdir -p "$OUT"

i=0
for pn in "$@"; do
    tag=$(printf "pn%s" "$pn")
    OMP_NUM_THREADS=1 "$BIN" --in "$CFG" --out "$OUT/$tag.cfg" \
        --log "$OUT/$tag.log" --dump "$OUT/$tag.dump" \
        --steps "$STEPS" --logevery 100 --dumpevery 2000 \
        --kT 1.0 --baro --plat 0.0 --pnorm "$pn" --taup 5.0 --kappa 0.1 \
        --ckptevery 20000 --seed $((2000 + i)) > "$OUT/$tag.out" 2>&1 &
    i=$((i + 1))
    if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
echo "pressure ladder complete: $OUT"
