#!/bin/bash
ffmpeg -s 352x288 -pixel_format gray -i ./files/foreman_cif_y_ME_recon.yuv ./files/E3/E3_%03d.bmp