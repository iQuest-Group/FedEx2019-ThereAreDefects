package com.iquest.fedex.therearedefects;

import com.iquest.fedex.therearedefects.service.ThereAreDefectsService;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@Slf4j
@SpringBootApplication
public class ThereAreDefectsApplication implements CommandLineRunner {

  private final ThereAreDefectsService service;

  @Autowired
  public ThereAreDefectsApplication(ThereAreDefectsService service) {
    this.service = service;
  }

  public static void main(String[] args) {
    SpringApplication.run(ThereAreDefectsApplication.class, args);
  }

  @Override
  public void run(String... args) throws Exception {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    String referenceInputFile = "C:\\Projects\\OpenCV\\Fedex-2019\\there-are-defects\\src\\main\\resources\\images\\Reference.png";
    Mat referenceObjectMeshMask = service.getMeshMaskInHSV(referenceInputFile, true, true, "Reference");

    String objectTestInputFile = "C:\\Projects\\OpenCV\\Fedex-2019\\there-are-defects\\src\\main\\resources\\images\\Defect1.png";
    Mat testObjectMeshMask = service.getMeshMaskInHSV(objectTestInputFile, true, true, "Defect1");

    boolean isDefect = service.isDefect(referenceObjectMeshMask, testObjectMeshMask);
    String status = isDefect ? "Is defect" : "Is the same";
    log.info(status);

    Mat testImage = Imgcodecs.imread(objectTestInputFile);
    service.drawDifferences(testImage, referenceObjectMeshMask, testObjectMeshMask);
  }
}
