package com.wellsfargo.ccibt.gplt.ftu.service;


import com.google.cloud.vertexai.VertexAI;
import com.google.cloud.vertexai.api.Blob;
import com.google.cloud.vertexai.api.Content;
import com.google.cloud.vertexai.api.Part;
import com.google.cloud.vertexai.generativeai.GenerativeModel;
import com.google.protobuf.ByteString;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class VertexAPIInvocationService {

    @Value("${vertexai.projectId}")
    private String projectId;
    @Value("${vertexai.location}")
    private String location;
    @Value("${vertexai.modelName}")
    private String modelName;
    @Value("${vertexai.queryPrompt}")
    private String queryPrompt;
    @Value("${vertexai.systemInstructions}")
    private String systemInstructions;

    String toHexString(String str) {
        StringBuilder hexString = new StringBuilder();
        for (char ch : str.toCharArray()) {
            hexString.append(String.format("%02x", (int) ch));
        }
        return hexString.toString();
    }

    public String classifyIncomingDocument(String content) throws Exception {

        Part systemInstructionsPart = Part.newBuilder().setText(systemInstructions).build();
        Content systemInstructionsContent = Content.newBuilder().addParts(systemInstructionsPart).build();


        String hexEmailContent = toHexString(content);
        try (VertexAI vertexAI = new VertexAI(projectId, location)) {
         GenerativeModel model = new GenerativeModel(modelName, vertexAI).withSystemInstruction(systemInstructionsContent);


            Part part1 = Part.newBuilder().setText(queryPrompt).build();

            Blob.Builder blobBuilder = Blob.newBuilder().setMimeType("text/plain").setData(ByteString.fromHex(hexEmailContent));
            Part part2 = Part.newBuilder().setInlineData(blobBuilder.build()).build();

            Content vertexAIContent = Content.newBuilder().setRole("user").addParts(part1).addParts(part2).build();

            String finalresponse = model.generateContentStream(vertexAIContent)
                    .stream()
                    .flatMap(response -> response.getCandidatesList().stream())
                    .flatMap(candidate -> candidate.getContent().getPartsList().stream())
                    .map(Part::getText).map(String::trim).toList().toString();
//                    .replace(", json", "{")
//                    .replace("[", "")
//                    .replace("]", "")
//                    .replace("`", "")
//                    .replace("{,", "");

            return finalresponse;
        }


    }
}
