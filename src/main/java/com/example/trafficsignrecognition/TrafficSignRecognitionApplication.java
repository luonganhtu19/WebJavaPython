package com.example.trafficsignrecognition;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@EnableAsync
@SpringBootApplication
public class TrafficSignRecognitionApplication {

    public static void main(String[] args) {
        SpringApplication.run(TrafficSignRecognitionApplication.class, args);
    }

}
