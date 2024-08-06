package main

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"gocv.io/x/gocv"
)

var output_shape = []int{640, 640}
var score_thresh = 0.75
var iou_thresh = 0.3
var labels = []string{"Fire"}

func preProcessing(imx gocv.Mat, height int64, width int64) (gocv.Mat, float64, float64, float64) {
	max_size := int64(math.Max(float64(height), float64(width)))
	scale_factor := float64(output_shape[0]) / float64(max_size)
	pad_h := (max_size - height) / 2
	pad_w := (max_size - width) / 2
	pad_image := gocv.NewMatWithSize(int(max_size), int(max_size), gocv.MatTypeCV8UC3)
	roi := pad_image.Region(image.Rect(int(pad_w), int(pad_h), int(pad_w+width), int(pad_h+height)))
	imx.CopyTo(&roi)
	resized := gocv.NewMat()
	gocv.Resize(pad_image, &resized, image.Pt(output_shape[0], output_shape[1]), 0, 0, gocv.InterpolationLinear)
	resized.ConvertTo(&resized, gocv.MatTypeCV32F)
	// resized.DivideFloat(255.0)
	return resized, scale_factor, float64(pad_h), float64(pad_w)
}

func postprocessing(outs gocv.Mat, img_width int64, img_height int64, scale_factor float64, pad_h float64, pad_w float64) ([]image.Rectangle, []int, []float32, string) {
	var scores []float32
	var bboxes []image.Rectangle
	var indices []int
	var label string
	min_conf := float32(0.3)
	cls_conf := float32(0.25)
	outs = outs.Reshape(0, outs.Size()[1])
	fmt.Println("Running")
	for i := 0; i < outs.Size()[0]; i++ {
		score1 := outs.GetFloatAt(i, 4) //Confidence
		score2 := outs.GetFloatAt(i, 5) //Class Confidence
		// fmt.Println(score1, score2)
		if score1 >= min_conf && score2 > cls_conf {
			scores = append(scores, score2)
			x1 := outs.GetFloatAt(i, 0)/float32(scale_factor) - float32(pad_w)
			x2 := outs.GetFloatAt(i, 1)/float32(scale_factor) - float32(pad_h)
			x3 := outs.GetFloatAt(i, 2) / float32(scale_factor)
			x4 := outs.GetFloatAt(i, 3) / float32(scale_factor)
			rect := image.Rect(
				int(x1-x3/2),
				int(x2-x4/2),
				int(x1+x3/2),
				int(x2+x4/2),
			)
			bboxes = append(bboxes, rect)
			label = "Fire"
		} else {
			continue
		}
	}
	if len(bboxes) > 0 {
		indices = gocv.NMSBoxes(bboxes, scores, float32(score_thresh), float32(iou_thresh))
	}
	return bboxes, indices, scores, label
}
func main() {
	model := gocv.ReadNetFromONNX("model.onnx")
	// fmt.Println(model)
	video, err := gocv.VideoCaptureFile("video.mp4")
	if err != nil {
		fmt.Println("Error opening video file:", err)
		return
	}
	defer video.Close()
	frameCounter := 0
	for {
		frame := gocv.NewMat()
		if ok := video.Read(&frame); !ok {
			fmt.Println("Cannot read frame from video file")
			break
		}
		if frame.Empty() {
			fmt.Println("Empty frame")
			break
		}
		frameCounter++
		if frameCounter < 100 {
			continue
		}
		img := frame.Clone()
		height, width := int64(img.Rows()), int64(img.Cols())
		gocv.CvtColor(img, &frame, gocv.ColorBGRToRGB)
		img, scale_factor, pad_h, pad_w := preProcessing(img, int64(img.Rows()), int64(img.Cols()))
		blob := gocv.BlobFromImage(img, 1.0/255.0, image.Pt(output_shape[0], output_shape[1]), gocv.Scalar{}, true, true)
		model.SetInput(blob, "images")
		outs := model.Forward("output0")
		var boxColor color.RGBA
		bboxes, indices, scores, label := postprocessing(outs, int64(height), int64(width), scale_factor, pad_h, pad_w)
		for _, ids := range indices {
			boxColor = color.RGBA{0, 255, 0, 255}
			gocv.Rectangle(&frame, bboxes[ids], boxColor, 2)
			labelText := fmt.Sprintf("%s: %f", label, scores[ids])
			labelPos := image.Pt(bboxes[ids].Min.X, bboxes[ids].Min.Y-10)
			gocv.PutText(&frame, labelText, labelPos, gocv.FontHersheyPlain, 1.2, boxColor, 2)
		}
		gocv.CvtColor(frame, &frame, gocv.ColorBGRToRGB)
		gocv.IMWrite("frame.png", frame)
		window := gocv.NewWindow("Video")
		window.IMShow(frame)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}
