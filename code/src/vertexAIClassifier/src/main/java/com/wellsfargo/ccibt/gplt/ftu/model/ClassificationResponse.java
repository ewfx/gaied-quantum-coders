package com.wellsfargo.ccibt.gplt.ftu.model;



import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class ClassificationResponse {
    private String       fileName;
    private String       type;
    private String       subtype;
    private String       summary;
    private Integer      confidence;
    private List<Entity> entities;
}
