#!/bin/bash

SCRIPT_DIR=/home/yaoyuh/Projects/dsta_mvs_datacollection/scripts/collect_for_comparison/omnimvs

cd $SCRIPT_DIR

python run_pre_processing_frame_graph.py \
	--data-dir /mnt/x/dsta_mvs_data/005_OmniMVSSunnySingle_frame_graph \
	--data-collection-metadata metadata.json \
	--data-collection-frame-graph frame_graph.json \
	--camera-config ../data/omnimvs/sunny_single.yaml \
	--out-dir  /mnt/x/dsta_mvs_data/005_OmniMVSSunnySingle_frame_graph/converted