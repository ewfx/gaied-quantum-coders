package com.wellsfargo.ccibt.gplt.ftu.controller;

import com.wellsfargo.ccibt.gplt.ftu.service.VertexAPIInvocationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("api/v2/emails")
public class EmailClassificationV2 {

    @Autowired
    private VertexAPIInvocationService vertexAPIInvocationService;
      @PostMapping("/classifyIncomingEmail")
         public String classifyIncomingEmail(@RequestBody String emailContent) throws Exception {
            return vertexAPIInvocationService.classifyIncomingDocument(emailContent);
      }



}
