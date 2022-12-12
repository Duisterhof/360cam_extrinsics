ECHO OFF

ECHO "OPENRESEARCHDRONE DATA COLLECTION PIPELINE VERSION 2.0.0"
python \\wsl$\Ubuntu-20.04\home\yaoyuh\Projects\dsta_mvs_datacollection\scripts\collect_for_comparison\run_collect_images_02.py ^
--metadata-path "metadata.json" ^
--requests-path "requests.json" ^
--frame-graph-path "frame_graph.json" ^
--image-width 2048 --image-height 1024
