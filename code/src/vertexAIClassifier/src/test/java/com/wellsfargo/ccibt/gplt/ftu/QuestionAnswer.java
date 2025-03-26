package com.wellsfargo.ccibt.gplt.ftu;

import com.google.cloud.vertexai.VertexAI;
import com.google.cloud.vertexai.api.GenerateContentResponse;
import com.google.cloud.vertexai.generativeai.GenerativeModel;
import com.google.cloud.vertexai.generativeai.ResponseHandler;

public class QuestionAnswer {

  public static void main(String[] args) throws Exception {
    // TODO(developer): Replace these variables before running the sample.
    String projectId = "glassy-filament-454505-q8";
    String location = "us-central1";
    String modelName = "gemini-1.5-flash-001";

    String output = simpleQuestion(projectId, location, modelName);
    System.out.println(output);
  }

  // Asks a question to the specified Vertex AI Gemini model and returns the generated answer.
  public static String simpleQuestion(String projectId, String location, String modelName)
      throws Exception {
    // Initialize client that will be used to send requests.
    // This client only needs to be created once, and can be reused for multiple requests.
    try (VertexAI vertexAI = new VertexAI(projectId, location)) {
      String output;
      GenerativeModel model = new GenerativeModel(modelName, vertexAI);
      // Send the question to the model for processing.
      GenerateContentResponse response = model.generateContent("How to become knowledgeable in Google Gen AI?");
      // Extract the generated text from the model's response.
      output = ResponseHandler.getText(response);
      return output;
    }
  }
}