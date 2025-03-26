package com.wellsfargo.ccibt.gplt.ftu.model;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter

public class EmailClassificationResponse {
    private ClassificationResponse emailClassification;
    private List<ClassificationResponse> attachmentClassifications;
}
