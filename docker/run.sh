#!/bin/bash
voila NWB_Panel.ipynb \
  --port=8866 \
  --Voila.ip=0.0.0.0 \
  --enable_nbextensions=True \
  --autoreload=True \
  --no-browser \
  --VoilaConfiguration.file_whitelist="['.*\.(png|jpg|gif|svg|mp4|avi|ico)']"
