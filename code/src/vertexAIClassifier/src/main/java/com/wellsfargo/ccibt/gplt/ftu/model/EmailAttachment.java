package com.wellsfargo.ccibt.gplt.ftu.model;


import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class EmailAttachment {
    private String fileName;
    private String fileType;
    private String location;
}
