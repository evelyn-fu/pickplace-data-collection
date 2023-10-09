cd $1/rgb
ffmpeg -framerate 33 -pattern_type glob -i "*.png"   -c:v libx264 -pix_fmt yuv420p ../video_out.mp4