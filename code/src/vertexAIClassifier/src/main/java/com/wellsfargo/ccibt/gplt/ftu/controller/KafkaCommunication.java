package com.wellsfargo.ccibt.gplt.ftu.controller;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("api/v1/kafka")
public class KafkaCommunication {

    @Autowired
    KafkaTemplate<String,String>    kafkaTemplate;

    @Value("${kafka.topic}")
    private String topic;

    @PostMapping("/sendMessage")
    ResponseEntity<String> sendMessage(String message){
        kafkaTemplate.send(topic,message);
        return ResponseEntity.ok("Message sent successfully");
    }
}
