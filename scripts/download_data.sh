#!/usr/bin/env bash
# download checkpoints
mkdir -p results;
sh ./scripts/gdrive_download.sh 1-E8786Y2OmgbP09_T6iX9Vq0bbC4c-WA  checkpoints.tar.gz;
tar -xvf checkpoints.tar.gz -C ./results/;
rm ./checkpoints.tar.gz

# download data
mkdir -p data;
sh ./scripts/gdrive_download.sh 1kBNP_PZ_MakYYBzj58WksFNg5U_c7yjS data.tar.gz;
tar -xvf data.tar.gz -C ./data/;
rm ./data.tar.gz

# download detections
mkdir -p detections;
sh ./scripts/gdrive_download.sh 1DQC7-ur-3czr5URD7LQF2o3p5j8Nu6AQ detections.tar.gz;
tar -xvf detections.tar.gz -C ./detections/;
rm ./detections.tar.gz

