package com.wellsfargo.ccibt.gplt.ftu.controller;

import com.wellsfargo.ccibt.gplt.ftu.service.VertexAPIInvocationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("api/v1/emails")
public class EmailClassification {

    @Autowired
    private VertexAPIInvocationService vertexAPIInvocationService;
      @PostMapping("/classifyIncomingEmail")
         public String classifyIncomingEmail(@RequestBody String emailContent) throws Exception {
            return vertexAPIInvocationService.classifyIncomingDocument(emailContent);
      }



}
