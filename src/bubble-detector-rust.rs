use opencv::{
    core::{Mat, Point, Vector},
    highgui, imgproc,
    prelude::*,
    types,
};
use std::time::Instant;

// BubbleDetector: Struct for detecting bubbles in an image
struct BubbleDetector {
    image: Mat,
    gray: Mat,
    blurred: Mat,
}

impl BubbleDetector {
    // Constructor: Create a new BubbleDetector instance
    fn new(image_path: &str) -> opencv::Result<Self> {
        let image = highgui::imread(image_path, highgui::IMREAD_COLOR)?;
        let gray = imgproc::cvt_color(&image, imgproc::COLOR_BGR2GRAY, 0)?;
        let blurred = imgproc::gaussian_blur(&gray, (3, 3), 1.0, 0.0, 0)?;
        Ok(Self { image, gray, blurred })
    }

    // detect_circles: Detect and draw circles on the image
    fn detect_circles(&mut self) -> opencv::Result<()> {
        let mut circles = Mat::default();
        imgproc::hough_circles(
            &self.blurred,
            &mut circles,
            imgproc::HOUGH_GRADIENT,
            1.0,
            10.0,
            100.0,
            30.0,
            1,
            25,
        )?;

        for circle in circles.iter::<types::Vec3f>()? {
            let center = Point::new(circle[0] as i32, circle[1] as i32);
            imgproc::circle(
                &mut self.image,
                center,
                circle[2] as i32,
                (0, 165, 255).into(),
                -1,
                imgproc::LINE_AA,
                0,
            )?;
        }
        Ok(())
    }

    // process_image: Detect and draw contours, measure performance
    fn process_image(&mut self) -> opencv::Result<()> {
        let start_time = Instant::now();

        let mut thresh = Mat::default();
        imgproc::threshold(&self.blurred, &mut thresh, 20.0, 255.0, imgproc::THRESH_BINARY)?;

        let mut contours = Vector::<Vector<Point>>::new();
        imgproc::find_contours(
            &thresh,
            &mut contours,
            imgproc::RETR_LIST,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        imgproc::draw_contours(
            &mut self.image,
            &contours,
            -1,
            (0, 165, 255).into(),
            -1,
            imgproc::LINE_8,
            &Mat::default(),
            0,
            Point::new(0, 0),
        )?;

        let end_time = Instant::now();
        println!("Bubble count: {}", contours.len());
        println!("Processing time: {:.2?}", end_time.duration_since(start_time));
        Ok(())
    }
}

fn main() -> opencv::Result<()> {
    let image_path = "image_bubbles.jpg";
    let mut detector = BubbleDetector::new(image_path)?;
    detector.detect_circles()?;
    detector.process_image()?;
    highgui::imshow("Processed Image", &detector.image)?;
    highgui::wait_key(0)?;
    Ok(())
}
