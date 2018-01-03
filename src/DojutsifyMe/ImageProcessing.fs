module DojutsifyMe.ImageProcessing

open Emgu.CV;

type GrayScaled = GrayScaled of Mat
type EqualizedHistogram = EqualizedHistogram of Mat

let grayScale frame = 
    let grayFrame = new Mat()
    CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray)
    GrayScaled grayFrame

let equalizeHistogram (GrayScaled frame) = 
    let equalizedFrame = new Mat()
    frame.CopyTo(equalizedFrame)
    CvInvoke.EqualizeHist(equalizedFrame, equalizedFrame)
    EqualizedHistogram equalizedFrame