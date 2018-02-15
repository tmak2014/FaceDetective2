package src;

import java.awt.AWTException;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.MouseInfo;
import java.awt.PointerInfo;
import java.awt.Robot;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class FdActivity extends javax.swing.JFrame {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCOEFF = 2;
    private static final int TM_CCOEFF_NORMED = 3;
    private static final int TM_CCORR = 4;
    private static final int TM_CCORR_NORMED = 5;

    private Point mBlackEyeCenter = null;
    private Point mSettingCenter = null;
    private double mDiffX = 0;
    private double mDiffY = 0;
    private List<Mat> mArrowImages = new ArrayList<>();
    
    /**
     * 1 2 3
     * 4 5 6
     * 7 8 9
     */
    private static final int DIRECTION_UP_RIGHT         = 1;
    private static final int DIRECTION_UP               = 2;
    private static final int DIRECTION_UP_LEFT          = 3;
    private static final int DIRECTION_RIGHT            = 4;
    private static final int DIRECTION_CENTER           = 5;
    private static final int DIRECTION_LEFT             = 6;
    private static final int DIRECTION_DOWN_RIGHT       = 7;
    private static final int DIRECTION_DOWN             = 8;
    private static final int DIRECTION_DOWN_LEFT        = 9;
    private int mEyeDirection = DIRECTION_CENTER;
    
    private int learn_frames = 0;
    private Mat teplateR;
    private Mat teplateL;
    
    private Mat dirMat;
    
    int method = TM_SQDIFF;
//case TM_SQDIFF:
//case TM_SQDIFF_NORMED:
//case TM_CCOEFF:
//case TM_CCOEFF_NORMED:
//case TM_CCORR:
//case TM_CCORR_NORMED:

    // matrix for zooming
    private Mat mZoomWindow;
    private Mat mZoomWindow2;

    // matrix for showing direction
    private Mat mDirectionWindow;

    private Mat                    mRgba = new Mat();
    private Mat                    mGray = new Mat();

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.5f;
    private int mAbsoluteFaceSize = 0;

    double xCenter = -1;
    double yCenter = -1;
    
    private final boolean _flag_mouse_control = true;

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        mZoomWindow.release();
        mZoomWindow2.release();
        mDirectionWindow.release();
    }

    public Mat onCameraFrame() {

        mRgba = frame;
        Imgproc.cvtColor(mRgba, mGray,Imgproc.COLOR_RGB2GRAY);

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }

        }

        if (mZoomWindow == null || mZoomWindow2 == null)
            CreateAuxiliaryMats();

        if (mDirectionWindow == null)
            CreateDirectionMat();

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else {
//            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
        {    Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(),
                FACE_RECT_COLOR, 3);
            xCenter = (facesArray[i].x + facesArray[i].width + facesArray[i].x) / 2;
            yCenter = (facesArray[i].y + facesArray[i].y + facesArray[i].height) / 2;
            Point center = new Point(xCenter, yCenter);

            Imgproc.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3);

            Imgproc.putText(mRgba, "[" + center.x + "," + center.y + "]",
                    new Point(center.x + 20, center.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));

            Rect r = facesArray[i];
            // compute the eye area
            Rect eyearea = new Rect(r.x + r.width / 8,
                    (int) (r.y + (r.height / 4.5)), r.width - 2 * r.width / 8,
                    (int) (r.height / 3.0));
            // split it
            Rect eyearea_right = new Rect(r.x + r.width / 16,
                    (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            Rect eyearea_left = new Rect(r.x + r.width / 16
                    + (r.width - 2 * r.width / 16) / 2,
                    (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            // draw the area - mGray is working grayscale mat, if you want to
            // see area in rgb preview, change mGray to mRgba
            Imgproc.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
                    new Scalar(255, 0, 0, 255), 2);
            Imgproc.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
                    new Scalar(255, 0, 0, 255), 2);

            if (learn_frames < 5) {
                teplateR = get_template(mJavaDetectorEyeR, eyearea_right, 20);
                teplateL = get_template(mJavaDetectorEyeL, eyearea_left, 20);
                learn_frames++;
            } else {
                // Learning finished, use the new templates for template
                // matching
                match_eye(eyearea_right, teplateR, method, 0);
                match_eye(eyearea_left, teplateL, method, 1);

            }


            // cut eye areas and put them to zoom windows
            Imgproc.resize(mRgba.submat(eyearea_left), mZoomWindow2,
                    mZoomWindow2.size());
            Imgproc.resize(mRgba.submat(eyearea_right), mZoomWindow,
                    mZoomWindow.size());

            Mat arrowImage = mArrowImages.get(mEyeDirection -1);
            Imgproc.resize(arrowImage, mDirectionWindow, mDirectionWindow.size());
        }

        return mRgba;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void CreateAuxiliaryMats() {
        if (mGray.empty())
            return;

        int rows = mGray.rows();
        int cols = mGray.cols();

        if (mZoomWindow == null) {
            mZoomWindow = mRgba.submat(rows / 2 + rows / 10, rows, cols / 2
                    + cols / 10, cols);
            mZoomWindow2 = mRgba.submat(0, rows / 2 - rows / 10, cols / 2
                    + cols / 10, cols);
        }

    }
    
    private void CreateDirectionMat() {
        if (mGray.empty())
            return;

        int rows = mGray.rows();
        int cols = mGray.cols();

        if (mDirectionWindow == null) {
            mDirectionWindow = mRgba.submat(rows - 100, rows, 0, 100);
        }

    }

    private Point centerR;
    private Point centerL;
    private void match_eye(Rect area, Mat mTemplate, int type, int lr) {
//        System.out.println("### match_eye()");
        Point matchLoc;
        Mat mROI = mGray.submat(area);
        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;
        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
            return ;
        }
        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        switch (type) {
            case TM_SQDIFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF);
                break;
            case TM_SQDIFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_SQDIFF_NORMED);
                break;
            case TM_CCOEFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF);
                break;
            case TM_CCOEFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_CCOEFF_NORMED);
                break;
            case TM_CCORR:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR);
                break;
            case TM_CCORR_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult,
                        Imgproc.TM_CCORR_NORMED);
                break;
        }

        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);
        // there is difference in matching methods - best match is max/min value
        if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
            matchLoc = mmres.minLoc;
        } else {
            matchLoc = mmres.maxLoc;
        }

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x,
                matchLoc.y + mTemplate.rows() + area.y);

        if (false) {
            Imgproc.rectangle(mRgba, matchLoc_tx, matchLoc_ty, new Scalar(255, 255, 0, 255));
        } else {
            Point pt = new Point();
//            Point pt = lr == 0 ? centerR : centerL;
            pt.x = matchLoc_tx.x + (matchLoc_ty.x - matchLoc_tx.x)/2;
            pt.y = matchLoc_tx.y + (matchLoc_ty.y - matchLoc_tx.y)/2;
            
            // とりあえず右目のみ
            if (lr == 0) {
                // 黒目の中心座標を随時格納
                mBlackEyeCenter = pt;
                
                // 初回だけ格納
                if (mSettingCenter == null) {
                    mSettingCenter = pt;
                }
            }
            
//            if (pt.x < area.width / 4 || pt.x > (area.width / 4) * 3 ||
//                pt.y < area.height / 4 || pt.y > (area.height / 4) * 3) {
//
//            } else {
                Imgproc.circle(mRgba, pt, (int)((matchLoc_ty.x - matchLoc_tx.x)/2), new Scalar(255, 255, 0, 255), -1);
//            }
        }
        Rect rec = new Rect(matchLoc_tx,matchLoc_ty);


    }

    private float value1 = 1.15f;
    private int value2 = 2;

    private Mat get_template(CascadeClassifier clasificator, Rect area, int size) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();
        clasificator.detectMultiScale(mROI, eyes, value1, value2,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT
                        | Objdetect.CASCADE_SCALE_IMAGE, new Size(size, size),
                new Size());

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x,
                    (int) (e.tl().y + e.height * 0.4), (int) e.width,
                    (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);


            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Imgproc.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 2);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            eye_template = new Rect((int) iris.x - size / 2, (int) iris.y
                    - size / 2, size, size);
            Imgproc.rectangle(mRgba, eye_template.tl(), eye_template.br(),
                    new Scalar(255, 0, 0, 255), 2);
            template = (mGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

    public void resetLearnFrames()
    {
        System.out.println("### resetLearnFrames()");
        learn_frames = 0;
        mSettingCenter = null;
        mBlackEyeCenter = null;
        mDiffX = 0;
        mDiffY = 0;
        mEyeDirection = DIRECTION_CENTER;
    }
    
    private DaemonThread myThread = null;
    int count = 0;
    int t5;
    VideoCapture webSource = null;
    Mat frame = new Mat();
    Mat gray_img = new Mat();
    Mat eye_img;
    MatOfByte mem = new MatOfByte();
    Mat circles = new Mat();
    int fps = 0;
    int cnt = 0;
    int oldcnt = 0;
    final double f = (1000 /Core.getTickFrequency());
    double startTime,nowTime, diffTime;
    double[] mData;
    double mRho;
    Point mPt = new Point();

    CascadeClassifier faceDetector = new CascadeClassifier(getClass().getResource("/haarcascade_eye_tree_eyeglasses.xml").getPath().substring(1));
    CascadeClassifier mJavaDetector = new CascadeClassifier(getClass().getResource("/lbpcascade_frontalface.xml").getPath().substring(1));
    CascadeClassifier mJavaDetectorEye = new CascadeClassifier(getClass().getResource("/haarcascade_lefteye_2splits.xml").getPath().substring(1));
    CascadeClassifier mJavaDetectorEyeL = new CascadeClassifier(getClass().getResource("/haarcascade_mcs_lefteye.xml").getPath().substring(1));
    CascadeClassifier mJavaDetectorEyeR = new CascadeClassifier(getClass().getResource("/haarcascade_mcs_righteye.xml").getPath().substring(1));
    MatOfRect faceDetections = new MatOfRect();

    class DaemonThread implements Runnable {

        protected volatile boolean runnable = false;

        @Override
        public void run() {
            synchronized (this) {
                startTime = Core.getTickCount();
                while (runnable) {
                    if (webSource.grab()) {
                        try {
                            webSource.retrieve(frame);
                            onCameraFrame();
                            Graphics g = jPanel1.getGraphics();

                            Imgcodecs.imencode(".bmp", frame, mem);
                            Image im = ImageIO.read(new ByteArrayInputStream(mem.toArray()));
                            BufferedImage buff = (BufferedImage) im;
//                            if (g.drawImage(buff, 0, 0, getWidth(), getHeight()-150 , 0, 0, buff.getWidth(), buff.getHeight(), null)) {
                            if (g.drawImage(buff, 0, 0, getWidth(), getHeight()-100 , 0, 0, buff.getWidth(), buff.getHeight(), null)) {
                               if (runnable == false) {
                                   System.out.println("Paused ..... ");
                                   this.wait();
                               }
                            }

                            //TODO Adjust the values
                            int x_threshold = 10;
                            int y_threshold = 10;
                            int x_lost = 100;
                            int y_lost = 100;

                            if(mSettingCenter != null && mBlackEyeCenter != null) {
                                mDiffX = mBlackEyeCenter.x - mSettingCenter.x;
                                mDiffY = mBlackEyeCenter.y - mSettingCenter.y;
                            }

                            if (Math.abs(mDiffX) > x_lost || Math.abs(mDiffY) > y_lost) {
                                System.out.println("lost... ");
                                mDiffX = 0;
                                mDiffY = 0;
                                mEyeDirection = DIRECTION_CENTER;
                                resetLearnFrames();
                            } else {
                                if(mDiffX > x_threshold) {
                                    if(mDiffY > y_threshold) {
                                        mEyeDirection = DIRECTION_DOWN_LEFT;
                                    } else if(mDiffY < -y_threshold) {
                                        mEyeDirection = DIRECTION_UP_LEFT;
                                    } else {
                                        mEyeDirection = DIRECTION_LEFT;
                                    }
                                } else if(mDiffX < -x_threshold) {
                                    if(mDiffY > y_threshold) {
                                        mEyeDirection = DIRECTION_DOWN_RIGHT;
                                    } else if(mDiffY < -y_threshold) {
                                        mEyeDirection = DIRECTION_UP_RIGHT;
                                    } else {
                                        mEyeDirection = DIRECTION_RIGHT;
                                    }
                                } else {
                                    if(mDiffY > y_threshold) {
                                        mEyeDirection = DIRECTION_DOWN;
                                    } else if(mDiffY < -y_threshold) {
                                        mEyeDirection = DIRECTION_UP;
                                    } else {
                                        mEyeDirection = DIRECTION_CENTER;
                                    }
                                }
                            }
                            if (_flag_mouse_control) {
                                moveMouse(mEyeDirection);
                            }

                            nowTime = Core.getTickCount();
                            diffTime = (int)((nowTime- startTime)*f);

                            if (diffTime >= 1000) {
                               startTime = nowTime;
                               fps = cnt - oldcnt;
                               oldcnt = cnt;
                            }

                            g.setColor(Color.RED);
                            g.drawString(String.valueOf(fps), 20, 20);
                            g.drawString(String.valueOf(value1), 20, 40);
                            g.drawString(String.valueOf(value2), 20, 60);

                            g.drawString("mSettingCenter: " + String.valueOf(mSettingCenter), 20, 80);
                            g.drawString("mBlackEyeCenter: " + String.valueOf(mBlackEyeCenter), 20, 100);
                            g.drawString("mDiffX: " + String.valueOf(mDiffX), 20, 120);
                            g.drawString("mDiffY: " + String.valueOf(mDiffY), 20, 140);
                            g.drawString("mEyeDirection: " +String.valueOf(mEyeDirection), 20, 160);

                            cnt++;
                        } catch (Exception ex) {
                            System.out.printf("Error %s", ex.toString());
                        }
                    }
                }
            }
        }
    }

    private void moveMouse(int direction) throws AWTException {
//            System.out.println("Number of Mouse Button : " + MouseInfo.getNumberOfButtons());

            PointerInfo pointerInfo = MouseInfo.getPointerInfo();
            System.out.println("Location of Mouse : " + pointerInfo.getLocation());

            Robot robot = new Robot();

            // マウスを移動させる
            int moveDiff = 5;
            int mousePointX = pointerInfo.getLocation().x;
            int mousePointY = pointerInfo.getLocation().y;
            switch (direction) {
                case DIRECTION_UP_RIGHT:
                    mousePointX -= moveDiff;
                    mousePointY -= moveDiff;
                    break;
                case DIRECTION_UP: 
                    mousePointY -= moveDiff;
                    break;
                case DIRECTION_UP_LEFT:
                    mousePointX += moveDiff;
                    mousePointY -= moveDiff;
                    break;
                case DIRECTION_RIGHT:
                    mousePointX -= moveDiff;
                    break;
                case DIRECTION_CENTER: break;
                case DIRECTION_LEFT:
                    mousePointX += moveDiff;
                    break;
                case DIRECTION_DOWN_RIGHT:
                    mousePointX -= moveDiff;
                    mousePointY += moveDiff;
                    break;
                case DIRECTION_DOWN: 
                    mousePointY += moveDiff;
                    break;
                case DIRECTION_DOWN_LEFT:
                    mousePointX += moveDiff;
                    mousePointY += moveDiff;
                    break;
            }
            robot.mouseMove(mousePointX, mousePointY);
    }

/////////
    /**
     * Creates new form FaceDetection
     */
    public FdActivity() {
        initComponents();
        System.out.println(FdActivity.class.getResource("/haarcascade_frontalface_alt.xml").getPath().substring(1));
        
        initImages();

    }

    private void initImages() {
        // TODO 画像パスを適切に
        mArrowImages = new ArrayList<Mat>();
        // Range is 0-8.
        mArrowImages.add(Imgcodecs.imread("1_up_right.png"));
        mArrowImages.add(Imgcodecs.imread("2_up.png"));
        mArrowImages.add(Imgcodecs.imread("3_up_left.png"));
        mArrowImages.add(Imgcodecs.imread("4_right.png"));
        mArrowImages.add(Imgcodecs.imread("5_center.png"));
        mArrowImages.add(Imgcodecs.imread("6_left.png"));
        mArrowImages.add(Imgcodecs.imread("7_down_right.png"));
        mArrowImages.add(Imgcodecs.imread("8_down.png"));
        mArrowImages.add(Imgcodecs.imread("9_down_left.png"));
        System.out.println(new File("1_up_right.png").getAbsoluteFile());
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        jPanel_direction = new javax.swing.JPanel();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jButtonV1U = new javax.swing.JButton();
        jButtonV1D = new javax.swing.JButton();
        jButtonV2U = new javax.swing.JButton();
        jButtonV2D = new javax.swing.JButton();
        jButtonLearn = new javax.swing.JButton();
//        jButtonSetCenter = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
//        jPanel1.setPreferredSize(new Dimension(1080, 780));
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
//            .addGap(0, 376, Short.MAX_VALUE)
            .addGap(0, 600, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout jPanel_direction_Layout = new javax.swing.GroupLayout(jPanel_direction);
        jPanel_direction.setLayout(jPanel_direction_Layout);
        jPanel_direction.setPreferredSize(new Dimension(300, 300));
        jPanel_direction_Layout.setHorizontalGroup(
                jPanel_direction_Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                .addGap(0, 0, Short.MAX_VALUE)
        );
        
        jPanel_direction_Layout.setVerticalGroup(
                jPanel_direction_Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
//              .addGap(0, 376, Short.MAX_VALUE)
                .addGap(0, 600, Short.MAX_VALUE)
        );

        jButton1.setText("Start");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton2.setText("Pause");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        jButtonV1U.setText("V1_Up");
        jButtonV1U.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonV1UActionPerformed(evt);
            }
        });
        jButtonV1D.setText("V1_Dn");
        jButtonV1D.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonV1DActionPerformed(evt);
            }
        });
        jButtonV2U.setText("V2_Up");
        jButtonV2U.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonV2UActionPerformed(evt);
            }
        });
        jButtonV2D.setText("V2_Dn");
        jButtonV2D.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonV2DActionPerformed(evt);
            }
        });
        jButtonLearn.setText("ResetLearn");
        jButtonLearn.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                resetLearnFrames();
            }
        });

//        jButtonSetCenter.setText("SetCenter");
//        jButtonSetCenter.addActionListener(new java.awt.event.ActionListener() {
//            public void actionPerformed(java.awt.event.ActionEvent evt) {
//                setCenter();
//            }
//        });
//        
//        jButtonPopUp.setText("PopUp");
//        jButtonPopUp.addActionListener(new java.awt.event.ActionListener() {
//            public void actionPerformed(java.awt.event.ActionEvent evt) {
//                popUp();
//            }
//        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(24, 24, 24)
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
//                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE)
//                .addComponent(jPanel_direction, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE)
//                .addComponent(jPanel_direction, 300, 300, 300)
                .addContainerGap())
            .addGroup(layout.createSequentialGroup()
                .addGap(100, 100, 100)
                .addComponent(jButton1)
                .addComponent(jButton2)
                .addGap(86, 86, 86)
//                .addComponent(jButtonSetCenter)
//                .addGap(86, 86, 86)
                .addComponent(jButtonLearn)
                .addGap(86, 86, 86)
//                .addComponent(jButtonPopUp)
//                .addGap(86, 86, 86)
                .addComponent(jButtonV1U)
                .addComponent(jButtonV1D)
                .addGap(86, 86, 86)
                .addComponent(jButtonV2U)
                .addComponent(jButtonV2D)
                .addContainerGap(100, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
//                .addComponent(jPanel_direction, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
//                .addComponent(jPanel_direction, 300, 300, 300)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButton1)
                    .addComponent(jButton2)
                    .addComponent(jButtonLearn)
//                    .addComponent(jButtonSetCenter)
                    .addComponent(jButtonV1U)
                    .addComponent(jButtonV1D)
                    .addComponent(jButtonV2U)
                    .addComponent(jButtonV2D))
//                .addComponent(jButtonPopUp)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        myThread.runnable = false;            // stop thread
        jButton2.setEnabled(false);   // activate start button
        jButton1.setEnabled(true);     // deactivate stop button

        webSource.release();  // stop caturing fron cam


    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButtonV1UActionPerformed(java.awt.event.ActionEvent evt) {
        value1 += 0.1f;
    }
    private void jButtonV1DActionPerformed(java.awt.event.ActionEvent evt) {
        value1 -= 0.1f;
    }
    private void jButtonV2UActionPerformed(java.awt.event.ActionEvent evt) {
        value2 += 1;
    }
    private void jButtonV2DActionPerformed(java.awt.event.ActionEvent evt) {
        value2 -= 1;
    }

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed

        webSource = new VideoCapture(0); // video capture from default cam
//      boolean setFps = webSource.set(Videoio.CAP_PROP_FPS, 30);
//        webSource.set(Videoio.CAP_PROP_FRAME_WIDTH, 160);
//        webSource.set(Videoio.CAP_PROP_FRAME_HEIGHT, 120);
//        webSource.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
//        webSource.set(Videoio.CAP_PROP_FRAME_HEIGHT, 960);

        myThread = new DaemonThread(); //create object of threat class
        Thread t = new Thread(myThread);
        t.setDaemon(true);
        myThread.runnable = true;
        t.start();                 //start thrad
        jButton1.setEnabled(false);  // deactivate start button
        jButton2.setEnabled(true);  //  activate stop button


    }//GEN-LAST:event_jButton1ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(FdActivity.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(FdActivity.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(FdActivity.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(FdActivity.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new FdActivity().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel_direction;
    
    private javax.swing.JButton jButtonV1U;
    private javax.swing.JButton jButtonV1D;

    private javax.swing.JButton jButtonV2U;
    private javax.swing.JButton jButtonV2D;

    private javax.swing.JButton jButtonLearn;
    
//    private javax.swing.JButton jButtonSetCenter;
//    private javax.swing.JButton jButtonPopUp;

    // End of variables declaration//GEN-END:variables
}
