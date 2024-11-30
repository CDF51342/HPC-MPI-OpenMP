mkdir TestFiles images_seq

echo "Converting images..."
convert highres.jpg TestFiles/in.ppm
convert highres.jpg TestFiles/in.pgm
echo "Finish convertion!"

echo "Creating images solution..."
make contrast_seq
./contrast_seq
mv out.pgm images_seq/out.pgm
mv out_hsl.ppm images_seq/out_hsl.ppm
mv out_yuv.ppm images_seq/out_yuv.ppm
rm ./contrast_seq
echo "Done images_seq/ !"
