import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.lang.reflect.Method;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class FacialRecognitionApp extends JFrame {
    // GUI Components
    private JLabel cameraScreen;
    private JButton startButton, registerButton, trainButton, clearButton;
    private JTextField nameField;
    private JTextArea logArea;
    private JLabel statusLabel, profileCountLabel, accuracyLabel;
    private JProgressBar registrationProgress;

    // OpenCV Components
    private org.opencv.videoio.VideoCapture capture;
    private CascadeClassifier faceCascade;
    // Use Object and reflection to avoid compile-time dependency on opencv_contrib
    private Object faceRecognizer = null;

    // Constants
    private static final String FACE_CASCADE = "haarcascade_frontalface_alt2.xml";
    private static final String FACES_DIR = "face_database";
    private static final int SAMPLES_PER_PERSON = 50;
    private static final int FACE_WIDTH = 200;
    private static final int FACE_HEIGHT = 200;
    private static final int CAPTURE_DELAY_MS = 100;
    private static final double CONFIDENCE_THRESHOLD = 50.0; // Lower is better

    // State Variables
    private volatile boolean cameraActive = false;
    private volatile boolean isRegistering = false;
    private String currentName = "";
    private AtomicInteger captureCount = new AtomicInteger(0);
    private long lastCaptureTime = 0;

    // Recognition Data
    private Map<Integer, String> labelToName = new HashMap<>();
    private Map<String, Integer> nameToLabel = new HashMap<>();
    private int nextLabel = 0;
    private boolean modelTrained = false;

    public static void main(String[] args) {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            JOptionPane.showMessageDialog(null,
                    "OpenCV native library not found!\nPlease install OpenCV and set java.library.path.",
                    "Error", JOptionPane.ERROR_MESSAGE);
            System.exit(1);
        }
        SwingUtilities.invokeLater(FacialRecognitionApp::new);
    }

    public FacialRecognitionApp() {
        initializeOpenCV();
        setupGUI();
        loadExistingDatabase();
    }

    private void initializeOpenCV() {
        // Load face cascade
        faceCascade = new CascadeClassifier();
        if (!faceCascade.load(FACE_CASCADE)) {
            if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
                JOptionPane.showMessageDialog(this,
                        "Face cascade file not found!\nPlace haarcascade xml in the working directory.",
                        "Error", JOptionPane.ERROR_MESSAGE);
                System.exit(1);
            }
        }

        // Try to load LBPHFaceRecognizer via reflection (opencv_contrib)
        try {
            Class<?> lbphClass = Class.forName("org.opencv.face.LBPHFaceRecognizer");
            Method createM = lbphClass.getMethod("create", int.class, int.class, int.class, int.class, double.class);
            faceRecognizer = createM.invoke(null, 1, 8, 8, 8, 100.0);
            log("LBPH Face Recognizer initialized (opencv_contrib detected).");
        } catch (ClassNotFoundException cnf) {
            log("opencv_contrib face module not found — recognition disabled. Add opencv_contrib to enable.");
            faceRecognizer = null;
        } catch (Exception ex) {
            log("Error initializing face recognizer: " + ex.getMessage());
            faceRecognizer = null;
        }

        new File(FACES_DIR).mkdirs();
        log("System initialized successfully");
    }

    private void setupGUI() {
        setTitle("Facial Recognition System v2.0");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1100, 800);
        setLocationRelativeTo(null);

        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            // ignore
        }

        JPanel mainPanel = new JPanel(new java.awt.BorderLayout(10, 10));
        mainPanel.setBorder(new EmptyBorder(15, 15, 15, 15));
        setContentPane(mainPanel);

        // Top Panel - Statistics
        JPanel statsPanel = new JPanel(new java.awt.GridLayout(1, 3, 10, 5));
        statsPanel.setBorder(BorderFactory.createTitledBorder("System Status"));

        statusLabel = new JLabel("Status: Ready", SwingConstants.CENTER);
        statusLabel.setFont(new Font("Arial", Font.BOLD, 14));
        statsPanel.add(statusLabel);

        profileCountLabel = new JLabel("Registered: 0", SwingConstants.CENTER);
        profileCountLabel.setFont(new Font("Arial", Font.PLAIN, 14));
        statsPanel.add(profileCountLabel);

        accuracyLabel = new JLabel("Recognition: N/A", SwingConstants.CENTER);
        accuracyLabel.setFont(new Font("Arial", Font.BOLD, 14));
        accuracyLabel.setForeground(new Color(0, 120, 0));
        statsPanel.add(accuracyLabel);

        mainPanel.add(statsPanel, java.awt.BorderLayout.NORTH);

        // Center Panel - Camera Feed
        cameraScreen = new JLabel("Camera feed will appear here", SwingConstants.CENTER);
        cameraScreen.setBorder(BorderFactory.createTitledBorder("Live Camera Feed"));
        cameraScreen.setBackground(Color.BLACK);
        cameraScreen.setOpaque(true);
        cameraScreen.setPreferredSize(new Dimension(800, 600));
        mainPanel.add(cameraScreen, java.awt.BorderLayout.CENTER);

        // Right Panel - Controls and Log
        JPanel rightPanel = new JPanel(new java.awt.BorderLayout(5, 5));

        JPanel controlPanel = new JPanel(new java.awt.GridBagLayout());
        controlPanel.setBorder(BorderFactory.createTitledBorder("Controls"));
        java.awt.GridBagConstraints gbc = new java.awt.GridBagConstraints();
        gbc.insets = new java.awt.Insets(5, 5, 5, 5);
        gbc.fill = java.awt.GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;

        gbc.gridx = 0; gbc.gridy = 0; gbc.gridwidth = 2;
        startButton = new JButton("Start Camera");
        startButton.setFont(new Font("Arial", Font.BOLD, 14));
        startButton.setBackground(new Color(100, 200, 100));
        startButton.setForeground(Color.WHITE);
        startButton.setFocusPainted(false);
        startButton.addActionListener(e -> toggleCamera());
        controlPanel.add(startButton, gbc);

        gbc.gridy = 1; gbc.gridwidth = 1;
        controlPanel.add(new JLabel("Name:"), gbc);

        gbc.gridx = 1;
        nameField = new JTextField(12);
        nameField.setFont(new Font("Arial", Font.PLAIN, 13));
        controlPanel.add(nameField, gbc);

        gbc.gridx = 0; gbc.gridy = 2; gbc.gridwidth = 2;
        registerButton = new JButton("Register New Face");
        registerButton.setFont(new Font("Arial", Font.BOLD, 13));
        registerButton.setBackground(new Color(70, 130, 180));
        registerButton.setForeground(Color.WHITE);
        registerButton.setEnabled(false);
        registerButton.setFocusPainted(false);
        registerButton.addActionListener(e -> startRegistration());
        controlPanel.add(registerButton, gbc);

        gbc.gridy = 3;
        registrationProgress = new JProgressBar(0, SAMPLES_PER_PERSON);
        registrationProgress.setStringPainted(true);
        registrationProgress.setString("Ready");
        controlPanel.add(registrationProgress, gbc);

        gbc.gridy = 4;
        trainButton = new JButton("Train Recognition Model");
        trainButton.setFont(new Font("Arial", Font.BOLD, 13));
        trainButton.setBackground(new Color(255, 165, 0));
        trainButton.setForeground(Color.WHITE);
        trainButton.setFocusPainted(false);
        trainButton.addActionListener(e -> trainModel());
        controlPanel.add(trainButton, gbc);

        gbc.gridy = 5;
        clearButton = new JButton("Clear All Profiles");
        clearButton.setFont(new Font("Arial", Font.BOLD, 13));
        clearButton.setBackground(new Color(200, 80, 80));
        clearButton.setForeground(Color.WHITE);
        clearButton.setFocusPainted(false);
        clearButton.addActionListener(e -> clearDatabase());
        controlPanel.add(clearButton, gbc);

        rightPanel.add(controlPanel, java.awt.BorderLayout.NORTH);

        logArea = new JTextArea(15, 30);
        logArea.setEditable(false);
        logArea.setFont(new Font("Consolas", Font.PLAIN, 11));
        JScrollPane scrollPane = new JScrollPane(logArea);
        scrollPane.setBorder(BorderFactory.createTitledBorder("System Log"));
        rightPanel.add(scrollPane, java.awt.BorderLayout.CENTER);

        mainPanel.add(rightPanel, java.awt.BorderLayout.EAST);

        setVisible(true);
        log("Facial Recognition System ready");
    }

    private void toggleCamera() {
        if (!cameraActive) {
            capture = new org.opencv.videoio.VideoCapture(0);

            if (!capture.isOpened()) {
                log("ERROR: Could not access camera");
                JOptionPane.showMessageDialog(this,
                        "Could not access camera!", "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            capture.set(org.opencv.videoio.Videoio.CAP_PROP_FRAME_WIDTH, 640);
            capture.set(org.opencv.videoio.Videoio.CAP_PROP_FRAME_HEIGHT, 480);

            cameraActive = true;
            startButton.setText("Stop Camera");
            startButton.setBackground(new Color(200, 80, 80));
            registerButton.setEnabled(true);
            statusLabel.setText("Status: Camera Active");
            log("Camera started successfully");

            new Thread(this::processCameraFeed).start();
        } else {
            cameraActive = false;
            startButton.setText("Start Camera");
            startButton.setBackground(new Color(100, 200, 100));
            registerButton.setEnabled(false);
            statusLabel.setText("Status: Camera Stopped");
            log("Camera stopped");

            if (capture != null) {
                capture.release();
            }
        }
    }

    private void processCameraFeed() {
        Mat frame = new Mat();
        Mat grayFrame = new Mat();

        while (cameraActive && capture.read(frame)) {
            if (frame.empty()) continue;

            Core.flip(frame, frame, 1);

            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(grayFrame, grayFrame);

            MatOfRect faces = new MatOfRect();
            faceCascade.detectMultiScale(grayFrame, faces, 1.1, 5,
                    0, new Size(100, 100), new Size(300, 300));

            Rect[] faceArray = faces.toArray();

            if (isRegistering) {
                handleRegistration(grayFrame, frame, faceArray);
            } else if (modelTrained && faceRecognizer != null) {
                recognizeFaces(grayFrame, frame, faceArray);
            } else {
                for (Rect face : faceArray) {
                    Imgproc.rectangle(frame, face.tl(), face.br(),
                            new Scalar(255, 255, 0), 2);
                }
            }

            displayFrame(frame);
        }
    }

    private void handleRegistration(Mat grayFrame, Mat colorFrame, Rect[] faces) {
        if (faces.length != 1) {
            String msg = faces.length == 0 ?
                    "No face detected - Please position your face" :
                    "Multiple faces detected - Only one person should be visible";

            Imgproc.putText(colorFrame, msg,
                    new Point(20, 40), Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.7, new Scalar(0, 0, 255), 2);
            return;
        }

        Rect faceRect = faces[0];
        long currentTime = System.currentTimeMillis();

        if (currentTime - lastCaptureTime >= CAPTURE_DELAY_MS) {
            int count = captureCount.get();
            if (count < SAMPLES_PER_PERSON) {
                Mat face = new Mat(grayFrame, faceRect);
                Mat resizedFace = new Mat();
                Imgproc.resize(face, resizedFace, new Size(FACE_WIDTH, FACE_HEIGHT));

                File personDir = new File(FACES_DIR, currentName);
                personDir.mkdirs();
                String filename = String.format("%s/%s/face_%03d.jpg",
                        FACES_DIR, currentName, count);
                Imgcodecs.imwrite(filename, resizedFace);

                captureCount.incrementAndGet();
                lastCaptureTime = currentTime;

                SwingUtilities.invokeLater(() -> {
                    registrationProgress.setValue(captureCount.get());
                    registrationProgress.setString(
                            String.format("Capturing: %d/%d", captureCount.get(), SAMPLES_PER_PERSON));
                });

                if (count % 10 == 0) {
                    log(String.format("Captured %d/%d samples for %s",
                            captureCount.get(), SAMPLES_PER_PERSON, currentName));
                }

                if (captureCount.get() >= SAMPLES_PER_PERSON) {
                    completeRegistration();
                }
            }
        }

        drawRegistrationOverlay(colorFrame, faceRect);
    }

    private void recognizeFaces(Mat grayFrame, Mat colorFrame, Rect[] faces) {
        if (faceRecognizer == null) return;

        for (Rect faceRect : faces) {
            Mat face = new Mat(grayFrame, faceRect);
            Mat resizedFace = new Mat();
            Imgproc.resize(face, resizedFace, new Size(FACE_WIDTH, FACE_HEIGHT));

            int[] label = new int[1];
            double[] confidence = new double[1];

            try {
                Method predictM = faceRecognizer.getClass().getMethod("predict", Mat.class, int[].class, double[].class);
                predictM.invoke(faceRecognizer, resizedFace, label, confidence);

                String name = "Unknown";
                Scalar color = new Scalar(0, 0, 255);

                if (confidence[0] < CONFIDENCE_THRESHOLD && labelToName.containsKey(label[0])) {
                    name = labelToName.get(label[0]);
                    color = new Scalar(0, 255, 0);
                }

                double accuracy = Math.max(0, 100 - confidence[0]);

                // create final copies for the lambda
                final String nameFinal = name;
                final double accuracyFinal = accuracy;
                final Scalar colorFinal = color;
                final Rect rectFinal = faceRect;
                final double confidenceFinal = confidence[0];

                SwingUtilities.invokeLater(() -> {
                    if (!"Unknown".equals(nameFinal)) {
                        accuracyLabel.setText(String.format("Recognition: %s (%.1f%%)", nameFinal, accuracyFinal));
                    } else {
                        accuracyLabel.setText("Recognition: Unknown");
                    }
                });

                drawRecognitionOverlay(colorFrame, rectFinal, nameFinal, confidenceFinal, colorFinal);

            } catch (NoSuchMethodException nsme) {
                log("Predict method unavailable on recognizer: " + nsme.getMessage());
            } catch (Exception e) {
                log("Recognition error: " + e.getMessage());
            }
        }
    }

    private void drawRegistrationOverlay(Mat frame, Rect rect) {
        Scalar color = new Scalar(255, 200, 0);

        Imgproc.rectangle(frame, rect.tl(), rect.br(), color, 3);

        int cornerLen = 30;
        Imgproc.line(frame, rect.tl(),
                new Point(rect.x + cornerLen, rect.y), color, 5);
        Imgproc.line(frame, rect.tl(),
                new Point(rect.x, rect.y + cornerLen), color, 5);

        Imgproc.line(frame, new Point(rect.x + rect.width, rect.y),
                new Point(rect.x + rect.width - cornerLen, rect.y), color, 5);
        Imgproc.line(frame, new Point(rect.x + rect.width, rect.y),
                new Point(rect.x + rect.width, rect.y + cornerLen), color, 5);

        Imgproc.line(frame, new Point(rect.x, rect.y + rect.height),
                new Point(rect.x + cornerLen, rect.y + rect.height), color, 5);
        Imgproc.line(frame, new Point(rect.x, rect.y + rect.height),
                new Point(rect.x, rect.y + rect.height - cornerLen), color, 5);

        Imgproc.line(frame, rect.br(),
                new Point(rect.x + rect.width - cornerLen, rect.y + rect.height), color, 5);
        Imgproc.line(frame, rect.br(),
                new Point(rect.x + rect.width, rect.y + rect.height - cornerLen), color, 5);

        String text = String.format("Capturing: %d/%d", captureCount.get(), SAMPLES_PER_PERSON);
        Imgproc.putText(frame, text,
                new Point(rect.x, rect.y - 10),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

        Imgproc.putText(frame, "Hold still and move head slightly",
                new Point(rect.x, rect.y + rect.height + 30),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

    private void drawRecognitionOverlay(Mat frame, Rect rect, String name,
                                        double confidence, Scalar color) {
        Imgproc.rectangle(frame, rect.tl(), rect.br(), color, 2);

        double accuracy = Math.max(0, 100 - confidence);
        String label = name.equals("Unknown") ?
                "Unknown" :
                String.format("%s (%.1f%%)", name, accuracy);

        int baseline[] = {0};
        Size textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX,
                0.8, 2, baseline);

        Imgproc.rectangle(frame,
                new Point(rect.x, rect.y - textSize.height - 10),
                new Point(rect.x + textSize.width + 10, rect.y),
                color, -1);

        Imgproc.putText(frame, label,
                new Point(rect.x + 5, rect.y - 5),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.8,
                new Scalar(255, 255, 255), 2);
    }

    private void startRegistration() {
        String name = nameField.getText().trim();

        if (name.isEmpty() || !name.matches("[a-zA-Z0-9_]+")) {
            JOptionPane.showMessageDialog(this,
                    "Please enter a valid name (letters, numbers, and underscores only)",
                    "Invalid Name", JOptionPane.WARNING_MESSAGE);
            return;
        }

        File personDir = new File(FACES_DIR, name);
        if (personDir.exists()) {
            int result = JOptionPane.showConfirmDialog(this,
                    "Profile '" + name + "' already exists. Overwrite?",
                    "Confirm", JOptionPane.YES_NO_OPTION);
            if (result != JOptionPane.YES_OPTION) {
                return;
            }
            deleteDirectory(personDir);
        }

        currentName = name;
        captureCount.set(0);
        isRegistering = true;
        lastCaptureTime = 0;

        registerButton.setEnabled(false);
        nameField.setEnabled(false);
        registrationProgress.setValue(0);
        registrationProgress.setString("Starting...");
        statusLabel.setText("Status: Registering " + name);

        log("Starting registration for: " + name);
        log("Please look at the camera and move your head slightly");
    }

    private void completeRegistration() {
        isRegistering = false;

        SwingUtilities.invokeLater(() -> {
            log("Registration complete for " + currentName);
            log("Total samples captured: " + SAMPLES_PER_PERSON);

            statusLabel.setText("Status: Registration Complete");
            nameField.setText("");
            nameField.setEnabled(true);
            registerButton.setEnabled(true);
            registrationProgress.setValue(SAMPLES_PER_PERSON);
            registrationProgress.setString("Complete!");

            JOptionPane.showMessageDialog(this,
                    "Registration complete for " + currentName + "!\n" +
                            "Please click 'Train Recognition Model' to enable recognition.",
                    "Success", JOptionPane.INFORMATION_MESSAGE);

            updateProfileCount();
        });
    }

    private void trainModel() {
        new Thread(() -> {
            if (faceRecognizer == null) {
                SwingUtilities.invokeLater(() ->
                        JOptionPane.showMessageDialog(this,
                                "Face recognizer not available.\nInstall opencv_contrib to enable training.",
                                "Unavailable", JOptionPane.WARNING_MESSAGE));
                log("Training skipped — recognizer unavailable.");
                return;
            }

            try {
                SwingUtilities.invokeLater(() -> {
                    statusLabel.setText("Status: Training model...");
                    trainButton.setEnabled(false);
                });

                log("Starting model training...");

                File facesDir = new File(FACES_DIR);
                File[] personDirs = facesDir.listFiles(File::isDirectory);

                if (personDirs == null || personDirs.length == 0) {
                    log("No profiles found to train");
                    SwingUtilities.invokeLater(() -> {
                        JOptionPane.showMessageDialog(this,
                                "No profiles found! Please register faces first.",
                                "Error", JOptionPane.WARNING_MESSAGE);
                        statusLabel.setText("Status: Ready");
                        trainButton.setEnabled(true);
                    });
                    return;
                }

                List<Mat> images = new ArrayList<>();
                List<Integer> labels = new ArrayList<>();

                labelToName.clear();
                nameToLabel.clear();
                nextLabel = 0;

                for (File personDir : personDirs) {
                    String name = personDir.getName();
                    int label = nextLabel++;

                    labelToName.put(label, name);
                    nameToLabel.put(name, label);

                    File[] imageFiles = personDir.listFiles((dir, filename) ->
                            filename.toLowerCase().endsWith(".jpg") ||
                                    filename.toLowerCase().endsWith(".png"));

                    if (imageFiles != null) {
                        for (File imageFile : imageFiles) {
                            Mat img = Imgcodecs.imread(imageFile.getAbsolutePath(),
                                    Imgcodecs.IMREAD_GRAYSCALE);
                            if (!img.empty()) {
                                Imgproc.resize(img, img, new Size(FACE_WIDTH, FACE_HEIGHT));
                                images.add(img);
                                labels.add(label);
                            }
                        }
                    }

                    log("Loaded " + (imageFiles != null ? imageFiles.length : 0) +
                            " samples for " + name);
                }

                if (images.isEmpty()) {
                    log("No valid training images found");
                    SwingUtilities.invokeLater(() -> {
                        statusLabel.setText("Status: Ready");
                        trainButton.setEnabled(true);
                    });
                    return;
                }

                MatOfInt labelsMat = new MatOfInt();
                labelsMat.fromList(labels);

                Method trainM = faceRecognizer.getClass().getMethod("train", List.class, MatOfInt.class);
                trainM.invoke(faceRecognizer, images, labelsMat);

                modelTrained = true;

                log("Training complete! Model ready for recognition.");
                log("Trained with " + images.size() + " images from " + labelToName.size() + " people");

                SwingUtilities.invokeLater(() -> {
                    statusLabel.setText("Status: Model Trained - Ready");
                    trainButton.setEnabled(true);
                    updateProfileCount();
                    JOptionPane.showMessageDialog(this,
                            "Model training complete!\nRecognition is now active.",
                            "Success", JOptionPane.INFORMATION_MESSAGE);
                });

            } catch (Exception e) {
                log("Training error: " + e.getMessage());
                e.printStackTrace();
                SwingUtilities.invokeLater(() -> {
                    statusLabel.setText("Status: Training Failed");
                    trainButton.setEnabled(true);
                    JOptionPane.showMessageDialog(this,
                            "Training failed: " + e.getMessage(),
                            "Error", JOptionPane.ERROR_MESSAGE);
                });
            }
        }).start();
    }

    private void loadExistingDatabase() {
        File facesDir = new File(FACES_DIR);
        File[] personDirs = facesDir.listFiles(File::isDirectory);

        if (personDirs != null && personDirs.length > 0) {
            log("Found " + personDirs.length + " existing profile(s)");
            log("Click 'Train Recognition Model' to enable recognition");
            updateProfileCount();
        } else {
            log("No existing profiles found");
        }
    }

    private void clearDatabase() {
        int choice = JOptionPane.showConfirmDialog(this,
                "Delete all profiles and training data?\nThis cannot be undone!",
                "Confirm", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);

        if (choice == JOptionPane.YES_OPTION) {
            deleteDirectory(new File(FACES_DIR));
            new File(FACES_DIR).mkdirs();

            labelToName.clear();
            nameToLabel.clear();
            modelTrained = false;
            nextLabel = 0;

            log("Database cleared");
            updateProfileCount();
            accuracyLabel.setText("Recognition: N/A");

            JOptionPane.showMessageDialog(this,
                    "All profiles deleted successfully.",
                    "Success", JOptionPane.INFORMATION_MESSAGE);
        }
    }

    private void updateProfileCount() {
        File facesDir = new File(FACES_DIR);
        File[] personDirs = facesDir.listFiles(File::isDirectory);
        int count = (personDirs != null) ? personDirs.length : 0;
        profileCountLabel.setText("Registered: " + count);
    }

    private void displayFrame(Mat frame) {
        BufferedImage img = matToBufferedImage(frame);
        SwingUtilities.invokeLater(() ->
                cameraScreen.setIcon(new ImageIcon(img)));
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = mat.channels() == 1 ?
                BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR;

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        mat.get(0, 0, data);

        return image;
    }

    private void deleteDirectory(File dir) {
        if (!dir.exists()) return;

        File[] files = dir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    deleteDirectory(file);
                } else {
                    file.delete();
                }
            }
        }
        dir.delete();
    }

    private void log(String message) {
        String timestamp = new SimpleDateFormat("HH:mm:ss").format(new Date());
        String formatted = "[" + timestamp + "] " + message;
        System.out.println(formatted);

        SwingUtilities.invokeLater(() -> {
            logArea.append(formatted + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }
}
