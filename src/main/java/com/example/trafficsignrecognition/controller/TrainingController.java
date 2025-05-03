package com.example.trafficsignrecognition.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

@RestController
public class TrainingController {

    private boolean isTraining = false;

    @GetMapping("/")
    public String index() {
        try {
            return new String(Files.readAllBytes(Paths.get("src/main/resources/static/index.html")));
        } catch (Exception e) {
            System.err.println("Error loading index.html: " + e.getMessage());
            return "Error loading page: " + e.getMessage();
        }
    }

    @PostMapping("/train")
    public String startTraining() {
        if (isTraining) {
            return "Training is already in progress.";
        }
        try {
            isTraining = true;
            // Đường dẫn tuyệt đối đến train.py
            Path pythonScriptPath = Paths.get("src/main/resources/static/train.py").toAbsolutePath();
            System.out.println("Running Python script: " + pythonScriptPath);
            ProcessBuilder pb = new ProcessBuilder("python", pythonScriptPath.toString(), "--train");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                    System.out.println("Python output: " + line);
                }
            }

            int exitCode = process.waitFor();
            isTraining = false;
            if (exitCode == 0) {
                System.out.println("Training completed successfully!");
                return "Training completed successfully!\n" + output.toString();
            } else {
                System.err.println("Training failed with exit code: " + exitCode);
                return "Training failed with exit code: " + exitCode + "\n" + output.toString();
            }
        } catch (Exception e) {
            isTraining = false;
            System.err.println("Error during training: " + e.getMessage());
            return "Error during training: " + e.getMessage();
        }
    }

    @GetMapping("/status")
    public String getTrainingStatus() {
        try {
            Path logPath = Paths.get("training_log.json");
            if (!Files.exists(logPath)) {
                return "{\"error\": \"Training log not found\"}";
            }
            String logContent = new String(Files.readAllBytes(logPath));
            return logContent;
        } catch (Exception e) {
            System.err.println("Error reading training log: " + e.getMessage());
            return "{\"error\": \"Unable to read training log: " + e.getMessage() + "\"}";
        }
    }

    @GetMapping("/predictions")
    public String getPredictions() {
        try {
            Path predictionsPath = Paths.get("predictions.json");
            if (!Files.exists(predictionsPath)) {
                return "{\"error\": \"Predictions not found\"}";
            }
            String predictionsContent = new String(Files.readAllBytes(predictionsPath));
            return predictionsContent;
        } catch (Exception e) {
            System.err.println("Error reading predictions: " + e.getMessage());
            return "{\"error\": \"Unable to read predictions: " + e.getMessage() + "\"}";
        }
    }

    @PostMapping("/predict")
    public String predict(@RequestParam("file") MultipartFile file) {
        try {
            // Lưu file tạm
            String uploadDir = "uploads/";
            Files.createDirectories(Paths.get(uploadDir));
            Path filePath = Paths.get(uploadDir, "uploaded_image.jpg");
            Files.copy(file.getInputStream(), filePath, StandardCopyOption.REPLACE_EXISTING);
            System.out.println("Saved uploaded file: " + filePath.toAbsolutePath());

            // Gọi Python
            Path pythonScriptPath = Paths.get("src/main/resources/static/train.py").toAbsolutePath();
            System.out.println("Running Python script: " + pythonScriptPath);
            ProcessBuilder pb = new ProcessBuilder("python", pythonScriptPath.toString(), "--predict", filePath.toAbsolutePath().toString());
            pb.redirectErrorStream(true);
            Process process = pb.start();

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                    System.out.println("Python output: " + line);
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                Path resultPath = Paths.get("prediction_result.json");
                if (!Files.exists(resultPath)) {
                    return "{\"error\": \"Prediction result not found\"}";
                }
                String result = new String(Files.readAllBytes(resultPath));
                System.out.println("Prediction result: " + result);
                return result;
            } else {
                System.err.println("Prediction failed with exit code: " + exitCode);
                return "{\"error\": \"Prediction failed with exit code: " + exitCode + "\n" + output.toString() + "\"}";
            }
        } catch (Exception e) {
            System.err.println("Error during prediction: " + e.getMessage());
            return "{\"error\": \"Error during prediction: " + e.getMessage() + "\"}";
        }
    }
}