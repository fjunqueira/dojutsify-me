module DojutsifyMe.Utils

open System.Drawing
open Emgu.CV
open Emgu.CV.Structure
open Emgu.CV.Util

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

let approximateContours (curve:Point[]) =
    let approximatedCurve = new VectorOfPoint();
    let epsilon = 0.05 * CvInvoke.ArcLength(new VectorOfPoint(curve), true)
    CvInvoke.ApproxPolyDP(new VectorOfPoint(curve), approximatedCurve, epsilon, true)
    approximatedCurve.ToArray()

let minEnclosingCircle (points:Point[]) = CvInvoke.MinEnclosingCircle(points |> Array.map (fun p -> PointF(float32 p.X, float32 p.Y)))