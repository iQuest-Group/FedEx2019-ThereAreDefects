package com.iquest.fedex.therearedefects.service;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
public class ThereAreDefectsService {

  private static final Scalar LOWER_GREEN_HSV_BOUNDARY = new Scalar(25, 10, 10);
  private static final Scalar UPPER_GREEN_HSV_BOUNDARY = new Scalar(100, 300, 300);

  public boolean isDefect(Mat referenceMeshMask, Mat objectTestMeshMask) {
    int differences = 0;
    for (int row = 0; row < referenceMeshMask.rows(); row++) {
      for (int col = 0; col < referenceMeshMask.cols(); col++) {
        if (referenceMeshMask.get(row, col)[0] != objectTestMeshMask.get(row, col)[0]) {
          differences++;
        }
      }
    }
    return differences * 100.0 / (double) (referenceMeshMask.rows() * referenceMeshMask.cols()) > 0.2;
  }

  public void drawDifferences(Mat image, Mat referenceObjectMask, Mat testObjectMask) {
    Mat differencesMask = new Mat();
    Core.bitwise_xor(referenceObjectMask, testObjectMask, differencesMask);
    Imgcodecs.imwrite("C:\\Projects\\OpenCV\\Fedex-2019\\there-are-defects\\src\\main\\resources\\images\\DiffMask.bmp", differencesMask);

    List<MatOfPoint> contours = new ArrayList<>();
    Imgproc.findContours(differencesMask, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
    List<MatOfPoint> filteredContours = contours.stream()
                                                .filter(contour -> Imgproc.contourArea(contour) > 10.0)
                                                .collect(Collectors.toList());
    Mat filteredDifferencesMask = new Mat(differencesMask.rows(), differencesMask.cols(), CvType.CV_8UC1);
    for (int i = 0; i < filteredContours.size(); i++) {
      Imgproc.drawContours(filteredDifferencesMask, filteredContours, i, new Scalar(255, 255, 255));
    }
    Imgcodecs.imwrite("C:\\Projects\\OpenCV\\Fedex-2019\\there-are-defects\\src\\main\\resources\\images\\DiffMaskFiltered.bmp",
                      filteredDifferencesMask);
/////////
    Mat dilateFilteredDifferencesMask = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
    int kernelSize = 9;
    Mat element = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                                                new Point(kernelSize, kernelSize));
    Imgproc.dilate(filteredDifferencesMask, dilateFilteredDifferencesMask, element);
    Imgcodecs.imwrite("C:\\Projects\\OpenCV\\Fedex-2019\\there-are-defects\\src\\main\\resources\\images\\DiffMaskFilteredDilate.bmp",
                      dilateFilteredDifferencesMask);

    contours = new ArrayList<>();
    Imgproc.findContours(dilateFilteredDifferencesMask, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
    filteredContours = contours.stream()
                               .filter(contour -> Imgproc.contourArea(contour) > 300.0)
                               .collect(Collectors.toList());
    MatOfPoint2f[] contoursPoly = new MatOfPoint2f[filteredContours.size()];
    Point[] centers = new Point[filteredContours.size()];
    float[][] radius = new float[filteredContours.size()][1];
    for (int i = 0; i < filteredContours.size(); i++) {
      contoursPoly[i] = new MatOfPoint2f();
      Imgproc.approxPolyDP(new MatOfPoint2f(filteredContours.get(i).toArray()), contoursPoly[i], 3, true);
      centers[i] = new Point();
      Imgproc.minEnclosingCircle(contoursPoly[i], centers[i], radius[i]);
    }
    Mat diffImage = image.clone();
    List<MatOfPoint> contoursPolyList = new ArrayList<>(contoursPoly.length);
    for (MatOfPoint2f poly : contoursPoly) {
      contoursPolyList.add(new MatOfPoint(poly.toArray()));
    }
    Scalar red = new Scalar(0, 0, 255);
    Scalar blue = new Scalar(255, 0, 0);
    for (int i = 0; i < filteredContours.size(); i++) {
      Imgproc.drawContours(diffImage, contoursPolyList, i, blue, 3);
      Imgproc.circle(diffImage, centers[i], (int) radius[i][0], red, 6);
    }
    Imgcodecs.imwrite("C:\\Projects\\OpenCV\\Fedex-2019\\there-are-defects\\src\\main\\resources\\images\\Diff.bmp", diffImage);
  }

  public Mat getMeshMaskInHSV(String inputFile, boolean writeMeshMaskToImage, boolean writeMeshMaskToFile, String imageLabel) throws IOException {
    byte[] sourceImage = FileUtils.readFileToByteArray(new File(inputFile));
    Mat srcImage = Imgcodecs.imdecode(new MatOfByte(sourceImage), Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR);

    Mat hsvSrcImage = new Mat();
    Imgproc.cvtColor(srcImage, hsvSrcImage, Imgproc.COLOR_RGB2HSV);

    Mat meshImage = new Mat();
    Mat meshMaskImage = new Mat();
    Core.inRange(hsvSrcImage, LOWER_GREEN_HSV_BOUNDARY, UPPER_GREEN_HSV_BOUNDARY, meshMaskImage);

    if (writeMeshMaskToImage) {
      srcImage.copyTo(meshImage, meshMaskImage);
      Imgcodecs.imwrite("C:\\Projects\\OpenCV\\Fedex-2019\\there-are-defects\\src\\main\\resources\\images\\" + imageLabel + "Mesh.bmp", meshImage);
      log.info("Image written");
    }

    if (writeMeshMaskToFile) {
      writeMeshMaskToFile(srcImage, meshMaskImage, imageLabel);
    }
    return meshMaskImage;
  }

  private void writeMeshMaskToFile(Mat srcImage, Mat meshMaskImage, String imageLabel) {
    Path path = Paths.get("C:\\Projects\\OpenCV\\Fedex-2019\\there-are-defects\\src\\main\\resources\\images\\" + imageLabel + "Mesh.txt");
    try (BufferedWriter writer = Files.newBufferedWriter(path)) {
      for (int i = 0; i < srcImage.rows(); i++) {
        StringBuilder line = new StringBuilder();
        for (int j = 0; j < srcImage.cols(); j++) {
          line.append(Arrays.toString(meshMaskImage.get(i, j)));
          line.append(" ");
        }
        writer.write(line.toString());
        writer.newLine();
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
