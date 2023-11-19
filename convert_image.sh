#!/bin/bash
ffmpeg -s 352x288 -pixel_format gray -i ./files/foreman_cif_y_recon4_3.0.yuv ./files/foreman/%03d.bmp