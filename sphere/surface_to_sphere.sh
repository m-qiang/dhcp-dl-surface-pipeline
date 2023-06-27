SURF_HEMI='right'
HEMI='rh'
WORKERS=8
DATA_SPLIT='train'

python extract_surface.py --surf_hemi=$SURF_HEMI --data_split=$DATA_SPLIT --device='cuda:0'

source $FREESURFER_HOME/SetUpFreeSurfer.sh

ls ./data/$DATA_SPLIT | parallel --jobs $WORKERS mris_smooth -n 3 -nw -seed 1234 ./data/$DATA_SPLIT/{}/$HEMI.white ./data/$DATA_SPLIT/{}/$HEMI.smoothwm

ls ./data/$DATA_SPLIT | parallel --jobs $WORKERS mris_inflate -seed 1234 ./data/$DATA_SPLIT/{}/$HEMI.smoothwm ./data/$DATA_SPLIT/{}/$HEMI.inflated

ls ./data/$DATA_SPLIT | parallel --jobs $WORKERS mris_sphere -seed 1234 ./data/$DATA_SPLIT/{}/$HEMI.inflated ./data/$DATA_SPLIT/{}/$HEMI.sphere