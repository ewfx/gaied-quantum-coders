package com.wellsfargo.ccibt.gplt.ftu;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ApplicationRunnerClass implements CommandLineRunner {

    public static void main(String[] args) {
        SpringApplication.run(ApplicationRunnerClass.class, args);
    }

    @Override
    public void run(String... args) throws Exception {
        // Logic to execute when the application starts
        System.out.println("Spring Boot Application has started.");
    }
}