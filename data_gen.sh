#!/bin/bash
for VARIABLE in $(seq 1 1 1)
do
    # conda activate dedalus
    # mpiexec -n 1 python3 rayleigh_benard.py --seed $VARIABLE
    # python -m dedalus merge_procs snapshots
    npzfilename="data/rb2d_ra1e6_s""$VARIABLE""_bpvv.npz"
    python convert_to_npz.py -f 'data/snapshots_s*.h5' -o $npzfilename

    # python plot_slices.py data/ss_2e6_1_norm/*.h5
    # bash create_video.sh

    # filedir="data_seed""$VARIABLE"
    # mkdir -p $filedir
    # mv -v frames $filedir
    # mv $npzfilename $filedir
    # mv out.mp4 $filedir
    # rm -r snapshots
done