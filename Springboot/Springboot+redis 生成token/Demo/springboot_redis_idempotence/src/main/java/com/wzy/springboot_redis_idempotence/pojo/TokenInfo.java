package com.wzy.springboot_redis_idempotence.pojo;

public class TokenInfo {

    private String token;

    private Long tokenCreatedDate;

    private Long tokenExpiryDate;

    private String isLogin;


    public String getToken() {
        return token;
    }

    public void setToken(String token) {
        this.token = token;
    }

    public Long getTokenCreatedDate() {
        return tokenCreatedDate;
    }

    public void setTokenCreatedDate(Long tokenCreatedDate) {
        this.tokenCreatedDate = tokenCreatedDate;
    }

    public Long getTokenExpiryDate() {
        return tokenExpiryDate;
    }

    public void setTokenExpiryDate(Long tokenExpiryDate) {
        this.tokenExpiryDate = tokenExpiryDate;
    }

    public String getIsLogin() {
        return isLogin;
    }

    public void setIsLogin(String isLogin) {
        this.isLogin = isLogin;
    }
}

