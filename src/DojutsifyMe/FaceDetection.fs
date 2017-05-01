module DojutsifyMe.FaceDetection

open Emgu.CV;
open Emgu.CV.Structure;
open System.Drawing;

let detectEyes face (ugray:UMat) =
    use faceRegion = new UMat(ugray, face)
    use eye = new CascadeClassifier("haarcascade_eye.xml")
    
    eye.DetectMultiScale(faceRegion, 1.1, 10, Size(20, 20)) |>
        Array.map (fun (eye:Rectangle) -> eye.Offset(Point(face.X, face.Y)); eye) |> 
        Array.toList

let detectFace (image:IInputArray) = 
    use ugray = new UMat()
    use face = new CascadeClassifier("haarcascade_frontalface_default.xml")
    CvInvoke.CvtColor(image, ugray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray)
    CvInvoke.EqualizeHist(ugray, ugray)

    face.DetectMultiScale(ugray, 1.1, 10, Size(20, 20)) |> 
        Array.map (fun face -> (face, detectEyes face ugray)) |> 
        Array.toList