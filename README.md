# Polychora
Python animation of uniform polychora in four and higher dimensions. 

![](Images/three.gif)

### Installation

```
git clone https://github.com/pkdoshinji/Polychora
sudo pip3 install numpy
sudo pip3 install matplotlib
```

### Use
First create a subdirectory /frames in the directory where you are running the script.

Then, simply run:
```
./polychora.py
```

To create a GIF, you will need ImageMagick.

Navigate to the frames directory and enter the following on the command line:
```
convert anim*.png <moviename.gif>
```

The GIF can be opened and displayed from your browser.
