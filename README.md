# dojutsify-me

### This is a WIP

The goal here is to create a "snapchat like" video filter that adds [dojutsu](http://naruto.wikia.com/wiki/D%C5%8Djutsu) to the user eyes.

### Here is what I've got working so far

![alt tag](https://github.com/fjunqueira/dojutsify-me/blob/master/demo.gif)

## TODO

- [x] Detect eyes
- [x] Track their position
- [ ] Locate the iris
- [ ] Replace it with some Dojutsu

No advanced techniques like facial landmark detection, ASM or AAM are being used. The code is built entirely with Emgucv (cross platform .Net wrapper to the OpenCV image processing library)

### Build instructions

1. Open the DojutsifyMe.fsproj and replace the following line with your EmguCV installation dir:

```
    <EmguCVDir>..\..\..\emgucv\libs</EmguCVDir>
```

2. Run `build.sh`
