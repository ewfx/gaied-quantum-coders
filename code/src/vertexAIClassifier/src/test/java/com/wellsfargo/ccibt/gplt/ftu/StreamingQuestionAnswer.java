package com.wellsfargo.ccibt.gplt.ftu;

import com.google.cloud.vertexai.VertexAI;
import com.google.cloud.vertexai.api.*;
import com.google.cloud.vertexai.generativeai.GenerativeModel;
import com.google.cloud.vertexai.api.Blob.Builder;
import com.google.protobuf.ByteString;

import java.util.Base64;
import java.util.stream.Collectors;

public class StreamingQuestionAnswer {

  public static void main(String[] args) throws Exception {
    // TODO(developer): Replace these variables before running the sample.
    String projectId = "glassy-filament-454505-q8";
    String location = "us-central1";
    String modelName = "gemini-1.5-flash-001";
    String emailContent = "\t\t\t\tXYZ Bank N.A.\n" +
            "\t\t\tLoan Agency Services\n" +
            "Date: 01-MAR-2025\n" +
            "TO: ABC Bank, NATIONAL ASSOCIATION\n" +
            "ATTN: SENTHAMARAI SELVI\n" +
            "FAX: 123-456-786\n" +
            "Re: ASDF MID-ATLANTIC LLC $171.3MM 11-4-2022, TERM LOAN A-2\n" +
            "\n" +
            "\t\tDescription: Facility Lender Share Adjustment\n" +
            "\n" +
            "BORROWER: ASDF MID-ATLANTIC LLC\n" +
            "DEAL NAME: ASDF MID-ATLANTIC LLC $171.3MM 11-4-2022\n" +
            "\n" +
            "Effective 01-MAR-2025, the Lender Shares of facility TERM LOAN A-2 have been adjusted.\n" +
            "Your share of the commitment was USD 5,518,249.19. It has been Increased to USD 5,542,963.55.\n" +
            "\n" +
            "For: ABC Bank, NA\n" +
            "\n" +
            "Reference: ASDF MID-ATLANTIC LLC $171.3MM 11-4-2022,\n" +
            "\n" +
            "If you have any question, please call the undersigned.\n" +
            "\n" +
            "************************COMMENT******************************************************\n" +
            "PLEASE FUND YOUR SHARE OF $24,714.36\n" +
            "\n" +
            "\n" +
            "Bank Name: XYZ Bank NA\n" +
            "ABA # 023456781\n" +
            "Account #: 0024487012\n" +
            "Account Name: LIQ CLO Operating Account\n" +
            "Ref: ASDF MID-ATLANTIC LLC\n" +
            "\n" +
            "*************************************************************************************\n" +
            "\n" +
            "Regards,\n" +
            "\n" +
            "ROBERTS SCOTT\n" +
            "Telephone #:\n" +
            "Fax #:\n" +
            "\n" +
            "XYZ Commercial Banking is a brand name of XYX Bank, N.A. Member FDIC\n" +
            "\n" +
            "\n" +
            "\n" +
            "\n" +
            "\n" +
            "March 01, 2025 - 09:39:15 AM\n" +
            "\n" +
            "\n" +
            "\n" +
            "\n" +
            "\n";





    streamingQuestion(projectId, location, modelName,toHexString(emailContent));
  }

  public static String toHexString(String str) {
    StringBuilder hexString = new StringBuilder();
    for (char ch : str.toCharArray()) {
      hexString.append(String.format("%02x", (int) ch));
    }
    return hexString.toString();
  }

  // Ask a simple question and get the response via streaming.
  public static void streamingQuestion(String projectId, String location, String modelName,String emailContent)
      throws Exception {
    // Initialize client that will be used to send requests.
    // This client only needs to be created once, and can be reused for multiple requests.
    try (VertexAI vertexAI = new VertexAI(projectId, location)) {
      GenerativeModel model = new GenerativeModel(modelName, vertexAI);
      Part part1 = Part.newBuilder().setText("From a commercial lending perspective classify the email content and provide  type,subtype and  confidence score out of 100 as a single json output. Ignore named entities in provided output result . ").build();
      Builder blobBuilder = Blob.newBuilder().setMimeType("text/plain").setData(ByteString.fromHex(emailContent));
      Part part2 = Part.newBuilder().setInlineData(blobBuilder.build()).build();
      Content content = Content.newBuilder().setRole("user").addParts(part1).addParts(part2).build();
      GenerateContentResponse generateContentResponse = model.generateContent(content).getDefaultInstanceForType();
       String finalresponse =    model.generateContentStream(content)
              .stream()
                      .flatMap(response ->response.getCandidatesList().stream())
                              .flatMap(candidate -> candidate.getContent().getPartsList().stream())
                                      .map(Part::getText).map(String::trim).toList().toString()
                                      .replace(", json","{")
                                      .replace("[","")
                                      .replace("]","")
                                      .replace("`","")
           .replace("{,","") ;

      System.out.println(finalresponse);
    }
  }
}