module DojutsifyMe.Utils

open System.Drawing
open Emgu.CV
open Emgu.CV.Structure

let mapTuple f (a,b) = (f a, f b)

let pointfToTuple (pointf:PointF) = (pointf.X, pointf.Y)

let tupleToPointf (x, y) = PointF(x, y)

let meanOfPoints (points:PointF list) = 
    points |> List.map pointfToTuple
        |> List.reduce (fun (accX, accY) (x, y) -> (accX + x, accY + y))
        |> mapTuple (fun point -> point / (points |> List.length |> float32))
        |> tupleToPointf

let minMaxLoc image =
    let mutable minVal = ref 0.0
    let mutable maxVal = ref 0.0
    let mutable minLoc = ref <| Point()
    let mutable maxLoc = ref <| Point()
    
    CvInvoke.MinMaxLoc(image, minVal, maxVal, minLoc, maxLoc)        

    (minVal.Value, maxVal.Value, minLoc.Value, maxLoc.Value)

let meanStdDev input = 
    let mutable mean = ref <| new MCvScalar()
    let mutable stdDev = ref <| new MCvScalar()

    CvInvoke.MeanStdDev(input, mean, stdDev, null)

    (mean.Value, stdDev.Value)

let rangeTreshold minValue maxValue (input:Mat) = 
    let output = input.Clone()
    CvInvoke.InRange(input, new ScalarArray(new MCvScalar(minValue)), new ScalarArray(new MCvScalar(maxValue)), output)
    CvInvoke.BitwiseNot(output, output)
    output