package com.wellsfargo.ccibt.gplt.ftu;

import java.io.IOException;

import com.google.cloud.language.v2.*;
import com.google.cloud.language.v2.Entity.Type;
import com.google.protobuf.DescriptorProtos;
import com.google.protobuf.InvalidProtocolBufferException;

public class AnalyzeEmail {

    public static void main(String[] args) {
        String text = "\t\t\t\tXYZ Bank N.A.\n" +
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
        analyzeEmail(text);
    }

    private static void analyzeEmail(String emailContent) {
        try (LanguageServiceClient languageServiceClient = LanguageServiceClient.create()) {
            // Create a document object
            Document doc = Document.newBuilder()
                                    .setContent(emailContent)
                                    .setType(Document.Type.PLAIN_TEXT)
                                    .build();
            ClassifyTextRequest request = ClassifyTextRequest.newBuilder().setDocument(doc).build();
            ClassifyTextResponse response =  languageServiceClient.classifyText(request);

            for (ClassificationCategory category : response.getCategoriesList()) {
                    System.out.printf("Category name : %s, Confidence : %.3f\n",category.getName(), category.getConfidence());
                }
            }catch (IOException e) {
            e.printStackTrace();
        }
    }
}