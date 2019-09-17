package com.wzy.article.pojo;

import java.io.Serializable;

/**
 * 统一返回数据实体类
 */
public class Result implements Serializable{

    private Boolean flag; //是否成功
    private String message;//消息
    private Object data;//返回数据

    public Result() {
    }

    public Result(Boolean flag, String message) {
        this.flag = flag;
        this.message = message;
    }

    public Result(Boolean flag, String message, Object data) {
        this.flag = flag;
        this.message = message;
        this.data = data;
    }

    public Boolean getFlag() {
        return flag;
    }

    public void setFlag(Boolean flag) {
        this.flag = flag;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Object getData() {
        return data;
    }

    public void setData(Object data) {
        this.data = data;
    }
}
