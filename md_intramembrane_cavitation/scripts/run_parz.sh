#!/bin/bash
# Interleaflet separation pressure Par(Z): a set of COM-distance restraint
# windows on a tensionless periodic bilayer.  Four windows run concurrently,
# one thread each, matching the four available cores.
#
# usage: run_parz.sh <equilibrated.cfg> <outdir> <equil_steps> <prod_steps> <kcom> [windows...]
set -u
CFG=$1; OUT=$2; EQ=$3; PROD=$4; KCOM=$5; shift 5
WINDOWS="$*"
BIN="$(dirname "$0")/../src/cgmd"
mkdir -p "$OUT"

TOTAL=$((EQ + PROD))
i=0
for dz in $WINDOWS; do
    tag=$(printf "dz%05.2f" "$dz")
    OMP_NUM_THREADS=1 "$BIN" --in "$CFG" --out "$OUT/$tag.cfg" \
        --log "$OUT/$tag.log" --dump "$OUT/$tag.dump" --steps "$TOTAL" \
        --logevery 100 --dumpevery 3000 \
        --kT 1.0 --comrestr --kcom "$KCOM" --dztarget "$dz" \
        --baro --plat 0.0 --pnorm 0.0 --taup 5.0 --kappa 0.1 \
        --softstart 500 --seed $((1000 + i)) > "$OUT/$tag.out" 2>&1 &
    i=$((i + 1))
    if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
echo "Par(Z) windows complete: $OUT"
